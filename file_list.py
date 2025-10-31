import os
import glob


def collect_python_files():
    """
    Collect all .py files in current directory and combine their contents
    into a single text file for sharing/analysis.
    """

    # Find all Python files in current directory
    python_files = glob.glob("*.py")
    python_files.sort()

    if not python_files:
        print("No Python files found in current directory!")
        return

    print(f"Found {len(python_files)} Python files:")
    for f in python_files:
        print(f"  - {f}")

    # Create combined output file
    output_file = "all_python_code.txt"

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("=" * 80 + "\n")
        outfile.write("ALL PYTHON FILES IN PROJECT\n")
        outfile.write("=" * 80 + "\n\n")

        for i, py_file in enumerate(python_files, 1):
            # Write header for each file
            outfile.write(f"\n{'='*60}\n")
            outfile.write(f"FILE {i}/{len(python_files)}: {py_file}\n")
            outfile.write(f"{'='*60}\n\n")

            try:
                # Read and write file contents
                with open(py_file, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)

                # Add separator
                outfile.write(f"\n\n{'#'*40} END OF {py_file} {'#'*40}\n\n")

            except Exception as e:
                outfile.write(f"ERROR: Could not read {py_file}: {e}\n\n")

    print(f"\n✅ All Python files combined into: {output_file}")
    print(f"You can now open {output_file} and copy-paste its contents!")


if __name__ == "__main__":
    collect_python_files()
