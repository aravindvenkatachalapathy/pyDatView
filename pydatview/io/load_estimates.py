"""Low-cost decoded-memory estimates for large result-file readers."""

import os
from glob import glob

import numpy as np


def _fast_decoded_bytes(path):
    if os.path.splitext(path)[1].lower() != '.outb':
        return max(0, os.path.getsize(path) * 2)
    with open(path, 'rb') as stream:
        file_id = np.fromfile(stream, dtype='<i2', count=1)
        if file_id.size != 1 or int(file_id[0]) not in (1, 2, 3, 4):
            return None
        if int(file_id[0]) == 4:
            np.fromfile(stream, dtype='<i2', count=1)
        n_channels = np.fromfile(stream, dtype='<i4', count=1)
        n_rows = np.fromfile(stream, dtype='<i4', count=1)
        if n_channels.size != 1 or n_rows.size != 1:
            return None
    return int(n_rows[0]) * (int(n_channels[0]) + 1) * 8


def _bladed_sensor_path(path):
    base, extension = os.path.splitext(path)
    if extension.lower() == '.$pj':
        return None
    if '$' in extension:
        return base + extension.replace('$', '%')
    return path


def _bladed_dataset_bytes(sensor_path, lazy):
    from pydatview.io.bladed_out_file import (
        isBinary,
        read_bladed_sensor_file,
    )

    info = read_bladed_sensor_file(sensor_path)
    rows = int(info['nMajor'])
    sections = int(info['nSections'])
    sensors = int(info['nSensors'])
    data_path = sensor_path.replace('%', '$')
    if rows == 0 and os.path.isfile(data_path):
        itemsize = np.dtype(info['Precision']).itemsize
        rows = os.path.getsize(data_path) // max(
            1,
            itemsize * sections * sensors,
        )
    if lazy and isBinary(data_path):
        # Time, virtual/table index, and one active plot channel.
        return rows * 24
    return rows * (
        sections * sensors * np.dtype(info['Precision']).itemsize + 16
    )


def estimate_decoded_load_bytes(path, file_format):
    """Estimate peak resident data from inexpensive FAST/Bladed headers."""
    format_name = getattr(file_format, 'name', '')
    try:
        if format_name == 'FAST output file':
            return _fast_decoded_bytes(path)
        if format_name == 'Bladed output file':
            base, extension = os.path.splitext(path)
            if extension.lower() == '.$pj':
                return sum(
                    _bladed_dataset_bytes(sensor_path, lazy=True)
                    for sensor_path in sorted(
                        glob(base + '.%[0-9][0-9]*')
                    )
                )
            sensor_path = _bladed_sensor_path(path)
            if sensor_path and os.path.isfile(sensor_path):
                return _bladed_dataset_bytes(sensor_path, lazy=False)
    except (OSError, ValueError, KeyError):
        return None
    return None
