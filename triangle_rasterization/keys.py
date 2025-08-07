from typing import Literal


def create_key_defines(
    *sequence: tuple[Literal["camera", "shell", "tile", "depth"], int],
) -> tuple[tuple[str, int], ...]:
    offset = 0
    defines = []

    # The sequence should have the most significant bits on the left.
    for key, length in reversed(sequence):
        if key == "depth" and length != 32:
            raise ValueError("Depth must have 32 bits.")

        upper = key.upper()
        defines.append((f"{upper}_BIT_OFFSET", offset))
        defines.append((f"{upper}_BIT_LENGTH", length))

        offset += length

    assert offset == 64
    return tuple(defines)
