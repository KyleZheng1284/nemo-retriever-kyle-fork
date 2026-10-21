# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize PDFium object lifetimes across renderer and retry threads."""

from functools import wraps
from threading import RLock

PDFIUM_PROCESS_LOCK = RLock()


def serialized_pdfium(function):
    """Keep creation, use and disposal under the same process-wide lock."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with PDFIUM_PROCESS_LOCK:
            return function(*args, **kwargs)

    return wrapped
