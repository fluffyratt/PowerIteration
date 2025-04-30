import numpy as np


def generate_and_save_matrix(size, filename):
    print(f"Generating matrix {size}x{size}...")
    matrix = np.random.randint(0, 10, size=(size, size), dtype=np.uint8)

    print(f"Saving to {filename}...")
    with open(filename, 'w') as f:
        f.write(f"{size}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Done: {filename}")


# Виклик функції:
generate_and_save_matrix(5000, "matrix_5000x5000.txt")
