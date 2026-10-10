"""Portable dependency adapter; original FK and geometry math remain exact bytes."""
def need(value, message):
    if not value:
        raise ValueError(message)

def science():
    import science as original_science
    return original_science
