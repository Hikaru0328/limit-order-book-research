"""Execute curated notebooks atomically with this Python interpreter."""
from pathlib import Path
import argparse
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Optional notebook stems")
    args = parser.parse_args()
    paths = sorted((ROOT / "studies/queue_depletion/notebooks").glob("*.ipynb"))
    unknown = set(args.names) - {p.stem for p in paths}
    if unknown:
        parser.error("Unknown notebook name")
    for path in paths:
        if args.names and path.stem not in args.names:
            continue
        notebook = nbformat.read(path, as_version=4)
        with tempfile.TemporaryDirectory() as directory:
            manager = KernelSpecManager(kernel_dirs=[directory])
            client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}}, record_timing=False)
            client.create_kernel_manager()
            client.km.kernel_spec_manager = manager
            client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client.execute()
        nbformat.validate(notebook)
        temporary = path.with_suffix(".ipynb.tmp")
        nbformat.write(notebook, temporary)
        temporary.replace(path)
        print("Executed " + path.name, flush=True)
if __name__ == "__main__":
    main()
