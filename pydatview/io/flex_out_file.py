import os
import numpy as np
import pandas as pd

try:
    from .file import (
        File,
        WrongFormatError,
        BrokenFormatError,
        OptionalImportError,
    )
except ImportError:
    File = dict

    class WrongFormatError(Exception):
        pass

    class BrokenFormatError(Exception):
        pass

    class OptionalImportError(Exception):
        pass


# --------------------------------------------------------------------------------}
# --- FLEX low-level reader
# --------------------------------------------------------------------------------{

def split_off_by_pattern(fid, pattern=b"\xff\xff\xff\xff", limit=2):
    found = 0
    read = dict()
    eof = False

    while (found < limit or limit == -1) and not eof:
        iread = read.setdefault(found, b"")
        current = fid.read(1)
        iread += current
        read[found] = iread

        ifound = pattern in iread
        if ifound:
            found += 1

        eof = current == b""

    return read


def ReadReflex(
    filename,
    dtype=np.float32,
    return_rescaled=True,
    ensure_time_in_data=True,
):
    """
    Read a Senvion/FLEX binary result file.

    Parameters
    ----------
    filename : str
        FLEX .res/.int file.
    dtype : numpy dtype
        Output floating-point dtype.
    return_rescaled : bool
        If True, return physical values.
        If False, return raw uint16-derived values together with
        scale and offset.
    ensure_time_in_data : bool
        If True and Tsim is missing, append a reconstructed Tsim channel.

    Returns
    -------
    data
    scale
    offset
    SensorNames
    SensorUnits
    SensorIDs
    SensorStatus
    time_info
    """

    with open(filename, "rb") as fid:

        # -----------------------------------------------------------------
        # Header
        # -----------------------------------------------------------------
        head = split_off_by_pattern(fid)

        # Example: 8620, often related to the following block length.
        other1 = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        value = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        if len(value) != 1:
            raise BrokenFormatError(
                "Could not read number of FLEX sensors from '{}'.".format(
                    filename
                )
            )

        nSensors = abs(int(value[0]))

        if nSensors <= 0:
            raise BrokenFormatError(
                "Invalid number of FLEX sensors: {}".format(nSensors)
            )

        # -----------------------------------------------------------------
        # Sensor IDs
        # -----------------------------------------------------------------
        SensorIDs = np.fromfile(
            fid,
            dtype=np.int32,
            count=nSensors,
        )

        if len(SensorIDs) != nSensors:
            raise BrokenFormatError(
                "Expected {} FLEX sensor IDs, found {}.".format(
                    nSensors,
                    len(SensorIDs),
                )
            )

        # -----------------------------------------------------------------
        # Number of time samples
        # -----------------------------------------------------------------
        value = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        if len(value) != 1:
            raise BrokenFormatError(
                "Could not read number of FLEX time steps."
            )

        nTimeSteps = int(value[0])

        if nTimeSteps <= 0:
            raise BrokenFormatError(
                "Invalid number of FLEX time steps: {}".format(
                    nTimeSteps
                )
            )

        # Additional/header information.
        other2 = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        # -----------------------------------------------------------------
        # Sensor names
        # -----------------------------------------------------------------
        value = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        if len(value) != 1:
            raise BrokenFormatError(
                "Could not read FLEX sensor-name block length."
            )

        lenNames = int(value[0])

        raw_names = (
            fid.read(lenNames)
            .strip(b"\x00")
            .split(b"\x00")
        )

        SensorNames = [
            item.decode("cp1252")
            for item in raw_names
        ]

        # -----------------------------------------------------------------
        # Sensor units
        # -----------------------------------------------------------------
        value = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        if len(value) != 1:
            raise BrokenFormatError(
                "Could not read FLEX unit block length."
            )

        lenUnits = int(value[0])

        raw_units = (
            fid.read(lenUnits)
            .strip(b"\x00")
            .split(b"\x00")
        )

        SensorUnits = [
            item.decode("cp1252")
            for item in raw_units
        ]

        # -----------------------------------------------------------------
        # Sensor status
        # -----------------------------------------------------------------
        value = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        if len(value) != 1:
            raise BrokenFormatError(
                "Could not read FLEX status block length."
            )

        lenStatus = int(value[0])

        raw_status = (
            fid.read(lenStatus)
            .strip(b"\x00")
            .split(b"\x00")
        )

        # Decode the status values as well.
        SensorStatus = [
            item.decode("cp1252")
            for item in raw_status
        ]

        # -----------------------------------------------------------------
        # Time information
        # -----------------------------------------------------------------
        other3 = np.fromfile(
            fid,
            dtype=np.int32,
            count=1,
        )

        time_values = np.fromfile(
            fid,
            dtype=np.float32,
            count=2,
        )

        if len(time_values) != 2:
            raise BrokenFormatError(
                "Could not read FLEX start time and time step."
            )

        time_start = float(time_values[0])
        time_step = float(time_values[1])

        time_info = {
            "start": time_start,
            "step": time_step,
            "number": nTimeSteps,
        }

        # -----------------------------------------------------------------
        # Data block
        #
        # At the end of the file:
        #
        #   offset : float32[nSensors]
        #   scale  : float32[nSensors]
        #   data   : uint16[nSensors * nTimeSteps]
        #
        # Each sensor therefore occupies:
        #
        #   8 bytes
        #
        # for offset + scale and:
        #
        #   2*nTimeSteps bytes
        #
        # for uint16 data.
        # -----------------------------------------------------------------
        bytes_from_end = nSensors * (
            8 + 2 * nTimeSteps
        )

        fid.seek(
            -bytes_from_end,
            os.SEEK_END,
        )

        offset = np.fromfile(
            fid,
            dtype=np.float32,
            count=nSensors,
        )

        scale = np.fromfile(
            fid,
            dtype=np.float32,
            count=nSensors,
        )

        if len(offset) != nSensors:
            raise BrokenFormatError(
                "Could not read FLEX offset array."
            )

        if len(scale) != nSensors:
            raise BrokenFormatError(
                "Could not read FLEX scale array."
            )

        offset = (
            offset
            .astype(dtype)
            .reshape(1, -1)
        )

        scale = (
            scale
            .astype(dtype)
            .reshape(1, -1)
        )

        # -----------------------------------------------------------------
        # Raw data
        # -----------------------------------------------------------------
        expected_values = nSensors * nTimeSteps

        raw_data = np.fromfile(
            fid,
            dtype=np.uint16,
            count=expected_values,
        )

        if raw_data.size != expected_values:
            raise BrokenFormatError(
                "FLEX data size mismatch in '{}'. "
                "Expected {} values, found {}.".format(
                    filename,
                    expected_values,
                    raw_data.size,
                )
            )

        data = (
            raw_data
            .astype(dtype)
            .reshape(nSensors, nTimeSteps)
            .T
        )

    # ---------------------------------------------------------------------
    # Validate metadata
    # ---------------------------------------------------------------------
    if len(SensorNames) != nSensors:
        raise BrokenFormatError(
            "FLEX sensor-name count mismatch: "
            "{} names for {} sensors.".format(
                len(SensorNames),
                nSensors,
            )
        )

    if len(SensorUnits) != nSensors:
        raise BrokenFormatError(
            "FLEX sensor-unit count mismatch: "
            "{} units for {} sensors.".format(
                len(SensorUnits),
                nSensors,
            )
        )

    if len(SensorStatus) != nSensors:
        raise BrokenFormatError(
            "FLEX sensor-status count mismatch: "
            "{} status values for {} sensors.".format(
                len(SensorStatus),
                nSensors,
            )
        )

    # ---------------------------------------------------------------------
    # Add reconstructed Tsim when requested
    # ---------------------------------------------------------------------
    if ensure_time_in_data:

        if "Tsim" not in SensorNames:

            SensorNames.append("Tsim")
            SensorIDs = np.append(
                SensorIDs,
                99999,
            )
            SensorUnits.append("s")
            SensorStatus.append("Time")

            index = np.arange(
                nTimeSteps,
                dtype=dtype,
            ).reshape(-1, 1)

            data = np.append(
                data,
                index,
                axis=1,
            )

            scale = np.append(
                scale,
                [[time_step]],
                axis=1,
            )

            offset = np.append(
                offset,
                [[time_start]],
                axis=1,
            )

    # ---------------------------------------------------------------------
    # Convert raw values to physical quantities
    #
    #     physical = raw * scale + offset
    # ---------------------------------------------------------------------
    if return_rescaled:
        data = data * scale + offset

    return (
        data,
        scale,
        offset,
        SensorNames,
        SensorUnits,
        SensorIDs,
        SensorStatus,
        time_info,
    )


# --------------------------------------------------------------------------------}
# --- FLEX / Senvion output file
# --------------------------------------------------------------------------------{

class FLEXOutFile(File):

    @staticmethod
    def defaultExtensions():
        return [".res", ".int"]

    @staticmethod
    def formatName():
        return "FLEX output file"

    def _read(self, **kwargs):
        """
        Read FLEX / Senvion binary output.

        Raw FLEX data are stored internally together with scale and offset.
        pyDatView receives physical values in _toDataFrame().
        """

        dtype = kwargs.pop(
            "dtype",
            np.float32,
        )

        output_time_name = kwargs.pop(
            "output_time_name",
            "Time",
        )

        # We handle reconstructed time ourselves below.
        ensure_time_in_data = kwargs.pop(
            "ensure_time_in_data",
            False,
        )

        try:
            (
                data,
                scale,
                offset,
                names,
                units,
                IDs,
                status,
                time_info,
            ) = ReadReflex(
                self.filename,
                dtype=dtype,
                return_rescaled=False,
                ensure_time_in_data=ensure_time_in_data,
            )

        except WrongFormatError:
            raise

        except BrokenFormatError:
            raise

        except Exception as error:
            raise BrokenFormatError(
                "Could not read FLEX/Senvion file '{}':\n{}".format(
                    self.filename,
                    error,
                )
            ) from error

        # -----------------------------------------------------------------
        # Store raw FLEX information
        # -----------------------------------------------------------------
        self.data = np.asarray(data)
        self.scale = np.asarray(scale)
        self.offset = np.asarray(offset)

        self.time_info = time_info

        if self.data.ndim != 2:
            raise BrokenFormatError(
                "Expected a 2-D FLEX data array, "
                "received shape {} for file '{}'.".format(
                    self.data.shape,
                    self.filename,
                )
            )

        self.nt = self.data.shape[0]
        self.nSensors = self.data.shape[1]

        names = list(names)
        units = list(units)
        IDs = list(IDs)
        status = list(status)

        if not (
            len(names)
            == len(units)
            == len(IDs)
            == len(status)
            == self.nSensors
        ):
            raise BrokenFormatError(
                "Inconsistent FLEX channel information for '{}':\n"
                "  data columns = {}\n"
                "  names        = {}\n"
                "  units        = {}\n"
                "  IDs          = {}\n"
                "  status       = {}".format(
                    self.filename,
                    self.nSensors,
                    len(names),
                    len(units),
                    len(IDs),
                    len(status),
                )
            )

        # -----------------------------------------------------------------
        # Time handling
        # -----------------------------------------------------------------
        self.time_col = "Tsim" in names

        if self.time_col:
            i_time = names.index("Tsim")
            names[i_time] = output_time_name
            self.time_index = i_time
        else:
            self.time_index = None

        self.sensors = {
            "Name": names,
            "Unit": units,
            "ID": IDs,
            "Status": status,
        }

        # -----------------------------------------------------------------
        # Reconstruct time if Tsim does not exist
        # -----------------------------------------------------------------
        if not self.time_col:
            try:
                t_start = float(
                    time_info["start"]
                )
                t_step = float(
                    time_info["step"]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:

                raise BrokenFormatError(
                    "FLEX file '{}' contains no Tsim channel and valid "
                    "time_info['start']/time_info['step'] could not "
                    "be found.".format(
                        self.filename
                    )
                ) from error

            self.time = (
                t_start
                + t_step
                * np.arange(
                    self.nt,
                    dtype=np.float64,
                )
            )

        else:
            self.time = None

    # -------------------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------------------
    def _rescaled_data(self):
        """
        Convert raw FLEX values to physical values.

            physical = raw * scale + offset
        """

        try:
            return (
                self.data
                * self.scale
                + self.offset
            )

        except ValueError as error:
            raise BrokenFormatError(
                "Cannot apply FLEX scaling:\n"
                "  data shape   = {}\n"
                "  scale shape  = {}\n"
                "  offset shape = {}".format(
                    self.data.shape,
                    self.scale.shape,
                    self.offset.shape,
                )
            ) from error

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------
    def __repr__(self):

        s = []

        s.append(
            "FLEX / Senvion output file"
        )

        s.append(
            "Filename: {}".format(
                self.filename
            )
        )

        s.append(
            "Samples: {}".format(
                self.nt
            )
        )

        s.append(
            "Channels: {}".format(
                self.nSensors
            )
        )

        s.append(
            "Explicit time channel: {}".format(
                self.time_col
            )
        )

        if (
            not self.time_col
            and self.time_info is not None
        ):
            s.append(
                "Time info: {}".format(
                    self.time_info
                )
            )

        return "\n".join(s)

    # -------------------------------------------------------------------------
    # pyDatView DataFrame
    # -------------------------------------------------------------------------
    def _toDataFrame(self):
        """
        Return FLEX data in pyDatView format.

        Normal channels are:

            Name_[unit]

        The FLEX sensor ID is only added if two channels would otherwise
        have identical names.
        """

        data = self._rescaled_data()

        names = list(
            self.sensors["Name"]
        )

        units = list(
            self.sensors["Unit"]
        )

        IDs = list(
            self.sensors["ID"]
        )

        clean_units = [
            str(unit)
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .strip()
            for unit in units
        ]

        # -----------------------------------------------------------------
        # Construct clean base names
        # -----------------------------------------------------------------
        base_columns = []

        for i, (name, unit) in enumerate(
            zip(names, clean_units)
        ):

            if (
                self.time_col
                and i == self.time_index
            ):
                base_columns.append(
                    "Time_[s]"
                )

            else:
                base_columns.append(
                    "{}_[{}]".format(
                        name,
                        unit,
                    )
                )

        # Count duplicate names.
        counts = {}

        for column in base_columns:
            counts[column] = (
                counts.get(column, 0) + 1
            )

        # -----------------------------------------------------------------
        # Add FLEX ID only if needed to resolve duplicates
        # -----------------------------------------------------------------
        sensor_columns = []

        for i, (
            name,
            sensor_id,
            unit,
            base,
        ) in enumerate(
            zip(
                names,
                IDs,
                clean_units,
                base_columns,
            )
        ):

            if (
                self.time_col
                and i == self.time_index
            ):
                sensor_columns.append(
                    "Time_[s]"
                )

            elif counts[base] == 1:
                sensor_columns.append(
                    base
                )

            else:
                sensor_columns.append(
                    "{}_{}_[{}]".format(
                        name,
                        sensor_id,
                        unit,
                    )
                )

        # -----------------------------------------------------------------
        # Explicit Tsim exists
        # -----------------------------------------------------------------
        if self.time_col:

            return pd.DataFrame(
                data=data,
                columns=sensor_columns,
            )

        # -----------------------------------------------------------------
        # No Tsim:
        # prepend reconstructed time
        # -----------------------------------------------------------------
        columns = (
            ["Time_[s]"]
            + sensor_columns
        )

        table_data = np.column_stack(
            (
                self.time,
                data,
            )
        )

        return pd.DataFrame(
            data=table_data,
            columns=columns,
        )