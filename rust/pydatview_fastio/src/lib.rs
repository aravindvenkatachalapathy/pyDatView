use byteorder::{LittleEndian, ReadBytesExt};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

const FILE_FMT_WITH_TIME: i16 = 1;
const FILE_FMT_WITHOUT_TIME: i16 = 2;
const FILE_FMT_NO_COMPRESS_WITHOUT_TIME: i16 = 3;
const FILE_FMT_CHAN_LEN_IN: i16 = 4;

fn read_u8_string<R: Read>(reader: &mut R, len: usize) -> PyResult<String> {
    let mut buf = vec![0_u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(String::from_utf8_lossy(&buf).trim().to_string())
}

fn read_vec_i32<R: Read>(reader: &mut R, n: usize) -> PyResult<Vec<i32>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(
            reader
                .read_i32::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    Ok(out)
}

fn read_vec_i16<R: Read>(reader: &mut R, n: usize) -> PyResult<Vec<i16>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(
            reader
                .read_i16::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    Ok(out)
}

fn read_vec_f32<R: Read>(reader: &mut R, n: usize) -> PyResult<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(
            reader
                .read_f32::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    Ok(out)
}

fn read_vec_f64<R: Read>(reader: &mut R, n: usize) -> PyResult<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(
            reader
                .read_f64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    Ok(out)
}

#[pyfunction]
fn read_fast_outb(py: Python<'_>, filename: &str) -> PyResult<(Py<PyArray2<f64>>, Py<PyDict>)> {
    let (data, file_id, desc, chan_names, chan_units, name) = py.detach(|| -> PyResult<_> {
        let path = Path::new(filename);
        let file = File::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut reader = BufReader::new(file);

        let file_id = reader
            .read_i16::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if ![
            FILE_FMT_WITH_TIME,
            FILE_FMT_WITHOUT_TIME,
            FILE_FMT_NO_COMPRESS_WITHOUT_TIME,
            FILE_FMT_CHAN_LEN_IN,
        ]
        .contains(&file_id)
        {
            return Err(PyValueError::new_err(format!(
                "FileID not supported {}. Is it a FAST binary file?",
                file_id
            )));
        }

        let len_name = if file_id == FILE_FMT_CHAN_LEN_IN {
            reader
                .read_i16::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))? as usize
        } else {
            10_usize
        };

        let num_out_chans = reader
            .read_i32::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            as usize;
        let nt = reader
            .read_i32::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(e.to_string()))? as usize;

        let (time_scl, time_off, time_out1, time_incr) = if file_id == FILE_FMT_WITH_TIME {
            let scl = reader
                .read_f64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let off = reader
                .read_f64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (scl, off, 0.0, 0.0)
        } else {
            let t0 = reader
                .read_f64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let dt = reader
                .read_f64::<LittleEndian>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (0.0, 0.0, t0, dt)
        };

        let (col_scl, col_off) = if file_id == FILE_FMT_NO_COMPRESS_WITHOUT_TIME {
            (vec![1.0_f32; num_out_chans], vec![0.0_f32; num_out_chans])
        } else {
            (
                read_vec_f32(&mut reader, num_out_chans)?,
                read_vec_f32(&mut reader, num_out_chans)?,
            )
        };

        let len_desc = reader
            .read_i32::<LittleEndian>()
            .map_err(|e| PyValueError::new_err(e.to_string()))? as usize;
        let desc = read_u8_string(&mut reader, len_desc)?;

        let mut chan_names = Vec::with_capacity(num_out_chans + 1);
        let mut chan_units = Vec::with_capacity(num_out_chans + 1);
        for _ in 0..(num_out_chans + 1) {
            chan_names.push(read_u8_string(&mut reader, len_name)?);
        }
        for _ in 0..(num_out_chans + 1) {
            let unit = read_u8_string(&mut reader, len_name)?;
            chan_units.push(unit.trim_matches(&['(', ')'][..]).to_string());
        }

        let packed_time = if file_id == FILE_FMT_WITH_TIME {
            Some(read_vec_i32(&mut reader, nt)?)
        } else {
            None
        };

        let npts = nt
            .checked_mul(num_out_chans)
            .ok_or_else(|| PyValueError::new_err("FAST binary size overflow"))?;
        let mut data = Array2::<f64>::zeros((nt, num_out_chans + 1));

        if file_id == FILE_FMT_NO_COMPRESS_WITHOUT_TIME {
            let packed = read_vec_f64(&mut reader, npts)?;
            for i in 0..nt {
                for j in 0..num_out_chans {
                    data[[i, j + 1]] = packed[i * num_out_chans + j];
                }
            }
        } else {
            let packed = read_vec_i16(&mut reader, npts)?;
            for i in 0..nt {
                for j in 0..num_out_chans {
                    let scl = col_scl[j] as f64;
                    let off = col_off[j] as f64;
                    data[[i, j + 1]] = if scl.is_nan() && off.is_nan() {
                        0.0
                    } else {
                        (packed[i * num_out_chans + j] as f64 - off) / scl
                    };
                }
            }
        }

        if let Some(time_values) = packed_time {
            for i in 0..nt {
                data[[i, 0]] = (time_values[i] as f64 - time_off) / time_scl;
            }
        } else {
            for i in 0..nt {
                data[[i, 0]] = time_out1 + time_incr * i as f64;
            }
        }

        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok((data, file_id, desc, chan_names, chan_units, name))
    })?;

    let info = PyDict::new(py);
    info.set_item("name", name)?;
    info.set_item("description", desc)?;
    info.set_item("fileID", file_id)?;
    info.set_item("attribute_names", PyList::new(py, chan_names)?)?;
    info.set_item("attribute_units", PyList::new(py, chan_units)?)?;

    Ok((data.into_pyarray(py).unbind(), info.unbind()))
}

#[pyfunction]
fn read_bladed_binary(
    py: Python<'_>,
    filename: &str,
    n_major: usize,
    n_sections: usize,
    n_sensors: usize,
    n_dimens: usize,
    is_float64: bool,
) -> PyResult<(Py<PyArray2<f64>>, usize)> {
    let (data, inferred_major) = py.detach(|| -> PyResult<_> {
        if n_dimens != 2 && n_dimens != 3 {
            return Err(PyValueError::new_err(format!(
                "Unsupported Bladed NDIMENS value: {}",
                n_dimens
            )));
        }
        if n_sections == 0 || n_sensors == 0 {
            return Err(PyValueError::new_err(
                "Bladed dimensions must include at least one section and one sensor",
            ));
        }

        let bytes = fs::read(filename).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let value_size = if is_float64 { 8 } else { 4 };
        let n_values = bytes.len() / value_size;
        if n_values == 0 {
            return Err(PyValueError::new_err(
                "Bladed binary file contains no values",
            ));
        }

        let values_per_step = if n_dimens == 3 {
            n_sections
                .checked_mul(n_sensors)
                .ok_or_else(|| PyValueError::new_err("Bladed dimensions overflow"))?
        } else {
            n_sensors
        };
        let inferred_major = if n_major == 0 {
            n_values / values_per_step
        } else {
            n_major
        };
        let expected_values = inferred_major
            .checked_mul(values_per_step)
            .ok_or_else(|| PyValueError::new_err("Bladed data size overflow"))?;
        if n_values < expected_values {
            return Err(PyValueError::new_err(format!(
                "Bladed binary file is too short: {} values found, {} expected",
                n_values, expected_values
            )));
        }

        let n_cols = if n_dimens == 3 {
            n_sections * n_sensors
        } else {
            n_sensors
        };
        let mut data = Array2::<f64>::zeros((inferred_major, n_cols));
        if is_float64 {
            for i in 0..inferred_major {
                for j in 0..n_cols {
                    let offset = (i * values_per_step + j) * value_size;
                    let mut chunk = [0_u8; 8];
                    chunk.copy_from_slice(&bytes[offset..offset + 8]);
                    data[[i, j]] = f64::from_le_bytes(chunk);
                }
            }
        } else {
            for i in 0..inferred_major {
                for j in 0..n_cols {
                    let offset = (i * values_per_step + j) * value_size;
                    let mut chunk = [0_u8; 4];
                    chunk.copy_from_slice(&bytes[offset..offset + 4]);
                    data[[i, j]] = f32::from_le_bytes(chunk) as f64;
                }
            }
        }

        Ok((data, inferred_major))
    })?;

    Ok((data.into_pyarray(py).unbind(), inferred_major))
}

#[pymodule]
fn pydatview_fastio(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_fast_outb, m)?)?;
    m.add_function(wrap_pyfunction!(read_bladed_binary, m)?)?;
    Ok(())
}
