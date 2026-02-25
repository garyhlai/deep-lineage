"""Download a pre-generated dataset from Google Drive and set up the run directory."""

import argparse
import sys
import shutil
import tempfile
import time
import zipfile
import tarfile
import gzip
import pickle
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import save_run_timestamp
from deep_lineage.schema import TrajectoryList


class DatasetDownloader:
    """Handles downloading and setting up pre-generated datasets."""

    # Required files for a complete dataset
    REQUIRED_FILES = [
        "tree_data.pkl.gz",
        "expression_data.pkl.gz",
        "dataset.h5ad",
        "trajectories.pkl.gz",
        "trajectories_train.pkl.gz",
        "trajectories_val.pkl.gz",
        "trajectories_test.pkl.gz",
    ]

    # Optional files that may be present
    OPTIONAL_FILES = [
        "trajectory_splits_report.json",
    ]

    def __init__(self, skip_validation: bool = False):
        self.skip_validation = skip_validation
        self.temp_dir = None
        self.run_dir = None

    def extract_file_id_or_url(self, url: str) -> str:
        """Extract Google Drive file ID or return URL as-is for gdown."""
        # If it's already just a file ID, return it
        if not url.startswith("http"):
            return f"https://drive.google.com/uc?id={url}"

        # Otherwise return the URL as-is, gdown can handle various formats
        return url

    def download_from_drive(self, url: str, output_path: Path) -> None:
        """Download file from Google Drive using gdown."""
        print("Downloading from Google Drive...")

        try:
            print("   Using gdown for download...")

            result = subprocess.run(
                ["gdown", url, "-O", str(output_path), "--fuzzy"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.stdout:
                for line in result.stdout.split("\n"):
                    if line.strip():
                        print(f"   gdown: {line.strip()}")

            if result.returncode != 0:
                print(f"   gdown stderr: {result.stderr}")
                print(f"   gdown stdout: {result.stdout}")
                raise RuntimeError(f"gdown failed with return code {result.returncode}")

            if not output_path.exists():
                raise RuntimeError("Download failed - output file not created")

            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            if file_size_mb < 0.001:  # Less than 1KB
                raise RuntimeError(
                    f"Download failed - file too small ({file_size_mb:.3f} MB). "
                    f"Please check that the file is shared with 'Anyone with the link' permission."
                )

            print(f"   Downloaded {file_size_mb:.1f} MB to {output_path.name}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Download timed out after 10 minutes")
        except FileNotFoundError:
            raise RuntimeError("gdown is not installed.")
        except Exception as e:
            if "gdown" in str(e) and "not found" in str(e):
                raise RuntimeError("gdown is not installed.")
            raise

    def extract_archive(self, archive_path: Path, extract_to: Path) -> Path:
        """Extract archive file and return the data directory path."""
        print(f"Extracting archive: {archive_path.name}")

        extract_to.mkdir(parents=True, exist_ok=True)

        try:
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)

            elif archive_path.suffix in [".gz", ".tar"] or ".tar." in archive_path.name:
                # Handle .tar.gz, .tgz, .tar files
                if ".tar" in archive_path.name:
                    mode = "r:gz" if ".gz" in archive_path.name else "r"
                    with tarfile.open(archive_path, mode) as tar_ref:
                        tar_ref.extractall(extract_to)
                else:
                    raise ValueError(
                        f"Cannot extract .gz file that is not a tar archive: {archive_path}"
                    )

            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to extract archive: {e}")

        # Find the actual data directory (might be nested)
        data_dir = self._find_data_directory(extract_to)
        print(f"   Extracted to: {data_dir}")

        return data_dir

    def _find_data_directory(self, extract_path: Path) -> Path:
        """Find the directory containing the actual data files."""
        # Check if files are directly in extract_path
        if (extract_path / "tree_data.pkl.gz").exists():
            return extract_path

        # Look for a subdirectory containing the data
        for item in extract_path.iterdir():
            if item.is_dir():
                if (item / "tree_data.pkl.gz").exists():
                    return item

        # If still not found, look deeper (max 2 levels)
        for item in extract_path.iterdir():
            if item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_dir() and (subitem / "tree_data.pkl.gz").exists():
                        return subitem

        raise ValueError(
            "Could not find data files in extracted archive. Expected files like 'tree_data.pkl.gz'"
        )

    def validate_dataset(self, data_dir: Path) -> Dict[str, Any]:
        """Validate the dataset structure and contents."""
        print("\nValidating dataset...")

        validation_report = {
            "required_files": {},
            "optional_files": {},
            "dataset_stats": {},
        }

        missing_files = []
        for filename in self.REQUIRED_FILES:
            file_path = data_dir / filename
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                validation_report["required_files"][filename] = {
                    "present": True,
                    "size_mb": round(file_size_mb, 2),
                }
                print(f"   {filename} ({file_size_mb:.1f} MB)")
            else:
                missing_files.append(filename)
                validation_report["required_files"][filename] = {"present": False}
                print(f"   {filename} (missing)")

        if missing_files:
            raise ValueError(f"Missing required files: {', '.join(missing_files)}")

        for filename in self.OPTIONAL_FILES:
            file_path = data_dir / filename
            if file_path.exists():
                validation_report["optional_files"][filename] = True
                print(f"   {filename} (optional, present)")

        if not self.skip_validation:
            print("\n   Loading trajectory data for validation...")
            try:
                traj_path = data_dir / "trajectories.pkl.gz"
                trajectories = TrajectoryList.load(str(traj_path))
                n_total = len(trajectories.trajectories)

                train_traj = TrajectoryList.load(
                    str(data_dir / "trajectories_train.pkl.gz")
                )
                val_traj = TrajectoryList.load(
                    str(data_dir / "trajectories_val.pkl.gz")
                )
                test_traj = TrajectoryList.load(
                    str(data_dir / "trajectories_test.pkl.gz")
                )

                n_train = len(train_traj.trajectories)
                n_val = len(val_traj.trajectories)
                n_test = len(test_traj.trajectories)

                validation_report["dataset_stats"] = {
                    "total_trajectories": n_total,
                    "train_trajectories": n_train,
                    "val_trajectories": n_val,
                    "test_trajectories": n_test,
                    "split_sum_matches": (n_train + n_val + n_test) == n_total,
                }

                print(
                    f"""
   Dataset Statistics:
      Total trajectories: {n_total}
      Train: {n_train} ({n_train / n_total * 100:.1f}%)
      Val:   {n_val} ({n_val / n_total * 100:.1f}%)
      Test:  {n_test} ({n_test / n_total * 100:.1f}%)
      Split integrity: {"Valid" if validation_report["dataset_stats"]["split_sum_matches"] else "Invalid"}"""
                )

                with gzip.open(data_dir / "tree_data.pkl.gz", "rb") as f:
                    tree = pickle.load(f)
                    validation_report["dataset_stats"]["tree_nodes"] = len(
                        tree.get("nodes", [])
                    )
                    print(f"      Tree nodes: {len(tree.get('nodes', []))}")

            except Exception as e:
                print(f"   Warning: Could not fully validate data: {e}")

        return validation_report

    def setup_run_directory(self, data_dir: Path) -> Path:
        """Set up the run directory structure in local_bucket/."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(f"local_bucket/run_{timestamp}")

        print(f"\nSetting up run directory: {run_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)

        for file_path in data_dir.iterdir():
            if file_path.is_file():
                dest_path = run_dir / file_path.name
                print(f"   Copying {file_path.name}")
                shutil.copy2(file_path, dest_path)

        save_run_timestamp(timestamp)
        print(f"   Set current run timestamp: {timestamp}")

        self.run_dir = run_dir
        return run_dir

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and Path(self.temp_dir).exists():
            print("\nCleaning up temporary files...")
            shutil.rmtree(self.temp_dir)

    def download_and_setup(
        self, url: Optional[str] = None, file_id: Optional[str] = None
    ) -> Path:
        """Main method to download and set up the dataset."""
        print(
            """
╔══════════════════════════════════════════════╗
║     Dataset Downloader & Setup Tool          ║
╚══════════════════════════════════════════════╝
"""
        )

        if url:
            download_url = self.extract_file_id_or_url(url)
        elif file_id:
            download_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            raise ValueError("Either --url or --file_id must be provided")

        try:
            self.temp_dir = tempfile.mkdtemp(prefix="dataset_download_")
            temp_path = Path(self.temp_dir)
            print(f"Using temporary directory: {temp_path}")

            archive_path = temp_path / "dataset.zip"
            self.download_from_drive(download_url, archive_path)

            extract_dir = temp_path / "extracted"
            data_dir = self.extract_archive(archive_path, extract_dir)

            validation_report = self.validate_dataset(data_dir)
            run_dir = self.setup_run_directory(data_dir)

            if not self.skip_validation:
                report_path = run_dir / "download_validation_report.json"
                with open(report_path, "w") as f:
                    json.dump(validation_report, f, indent=2, default=str)

            print(
                f"""
╔══════════════════════════════════════════════╗
║            Setup Complete.                   ║
╚══════════════════════════════════════════════╝

Data location: {run_dir}
Dataset ready with:
   - {validation_report.get("dataset_stats", {}).get("total_trajectories", "N/A")} trajectories
   - Pre-split train/val/test sets
   - Tree and expression data

Next steps:
   1. Verify splits (recommended):
      uv run python scripts/data/verify_split_integrity.py

   2. Train model:
      uv run python scripts/synthetic_training/train_autoencoder.py

   3. Continue with pipeline...
"""
            )

            return run_dir

        except Exception as e:
            print(f"\nError: {e}")
            raise
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Download and set up pre-generated dataset from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Examples:
  # Download with Google Drive shareable link
  python scripts/download_and_set_up_data.py --url "https://drive.google.com/file/d/ABC123/view?usp=sharing"
  
  # Download with file ID only
  python scripts/download_and_set_up_data.py --file_id "1a2b3c4d5e6f7g8h9i0j"
  
  # Skip validation for faster setup
  python scripts/download_and_set_up_data.py --url "..." --skip_validation

Note: The file must be shared with 'Anyone with the link' permission on Google Drive.
After setup, the data will be available in local_bucket/run_TIMESTAMP/ and ready for use.
        """,
    )

    parser.add_argument("--url", type=str, help="Google Drive shareable link URL")
    parser.add_argument(
        "--file_id", type=str, help="Google Drive file ID (alternative to --url)"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip dataset validation for faster setup",
    )

    args = parser.parse_args()

    if not args.url and not args.file_id:
        parser.error("Either --url or --file_id must be provided")

    if args.url and args.file_id:
        print("Warning: Both --url and --file_id provided, using --url")

    try:
        subprocess.run(
            ["gdown", "--version"], capture_output=True, check=True, timeout=2
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        print("\nError: gdown is not installed.")
        sys.exit(1)

    downloader = DatasetDownloader(skip_validation=args.skip_validation)

    try:
        downloader.download_and_setup(url=args.url, file_id=args.file_id)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
