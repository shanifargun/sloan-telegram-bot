"""Compatibility module for imghdr functionality removed in Python 3.11."""
import os
import typing

def what(file: typing.Union[str, os.PathLike, typing.BinaryIO], h: typing.Optional[bytes] = None) -> typing.Optional[str]:
    """Simple implementation of imghdr.what"""
    if h is None:
        if isinstance(file, (str, os.PathLike)):
            with open(file, 'rb') as f:
                h = f.read(32)
        else:
            h = file.read(32)
            file.seek(0)
    
    if h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if h.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
        return 'gif'
    if h.startswith(b'BM'):
        return 'bmp'
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'
    
    return None
