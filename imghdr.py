"""
Recognize image file formats based on their first few bytes.

This is a simplified version of the imghdr module from the Python standard library,
created to provide compatibility for Streamlit.
"""

__all__ = ["what"]

def what(file, h=None):
    """
    Recognize the type of an image file.

    Parameters
    ----------
    file : str or file object
        The file to check.
    h : bytes
        The first few bytes of the file, if already read.
        If not provided, the first few bytes will be read from the file.

    Returns
    -------
    str or None
        The image type if recognized, else None.
    """
    if h is None:
        if isinstance(file, str):
            with open(file, 'rb') as f:
                h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
            
    if not h:
        return None

    if h.startswith(b'\xff\xd8'):
        return 'jpeg'
    if h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
        return 'gif'
    if h.startswith(b'BM'):
        return 'bmp'
    if h.startswith(b'\x49\x49\x2A\x00') or h.startswith(b'\x4D\x4D\x00\x2A'):
        return 'tiff'
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'
    if h.startswith(b'\x00\x00\x01\x00'):
        return 'ico'
    
    return None

# Tests
tests = [
    (b'\xff\xd8', 'jpeg'),
    (b'\x89PNG\r\n\x1a\n', 'png'),
    (b'GIF87a', 'gif'),
    (b'GIF89a', 'gif'),
    (b'BM', 'bmp'),
    (b'\x49\x49\x2A\x00', 'tiff'),
    (b'\x4D\x4D\x00\x2A', 'tiff'),
    (b'RIFF\x00\x00\x00\x00WEBP', 'webp'),
    (b'\x00\x00\x01\x00', 'ico'),
]

if __name__ == "__main__":
    for data, expected in tests:
        result = what(None, data)
        print(f"Testing {data[:10]!r}: got {result}, expected {expected}")
        assert result == expected