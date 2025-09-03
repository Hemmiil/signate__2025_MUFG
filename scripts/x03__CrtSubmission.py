import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

class F01__CrtSubmission:
    """
    A class to create submission data based on scripts/README.md.
    """
    def __init__(self, path: str, memo: str):
        self.path = path
        self.rawdata = None
        self.submitdata = None
        self.memo = memo
        self.timestamp = self.f02__timestamp()

    def f01__get_data(self):
        """Loads the .npy file from the specified path."""
        try:
            self.rawdata = np.load(self.path)
            print(f"Data loaded from {self.path} successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def f02__timestamp(self) -> str:
        """Returns the current timestamp in mmdd_hhMM format."""
        return datetime.now().strftime("%m%d_%H%M")

    def f03__crt_submission(self):
        """Creates the submission DataFrame."""
        sample_submit_path = "data/sample_submit.csv"
        try:
            sample_df = pd.read_csv(sample_submit_path, header=None)
        except FileNotFoundError:
            print(f"Error: Sample submission file not found at {sample_submit_path}")
            sys.exit(1)

        if len(self.rawdata) != len(sample_df):
            print(f"Error: Raw data length ({len(self.rawdata)}) does not match sample submission length ({len(sample_df)}).")
            sys.exit(1)

        self.submitdata = pd.DataFrame({
            'id': sample_df[0],
            'state': self.rawdata.flatten() # Assuming rawdata is 1D or can be flattened
        })
        print("Submission data created.")

    def f04__save_submission(self):
        """Saves the submission data and memo."""
        output_dir = "output/99_Submision"
        os.makedirs(output_dir, exist_ok=True)

        # Save submission CSV
        submission_filename = f"{self.timestamp}__submit.csv"
        submission_filepath = os.path.join(output_dir, submission_filename)
        self.submitdata.to_csv(submission_filepath, index=False, header=None)
        print(f"Submission saved to {submission_filepath}")

        # Save memo JSON
        memo_filepath = os.path.join(output_dir, "000__memo.json")
        memos = {}
        if os.path.exists(memo_filepath):
            with open(memo_filepath, 'r') as f:
                try:
                    memos = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode existing memo file {memo_filepath}. Starting with empty memos.")

        memos[self.timestamp] = self.memo
        with open(memo_filepath, 'w') as f:
            json.dump(memos, f, indent=2, ensure_ascii=False)
        print(f"Memo saved to {memo_filepath}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python x03__CrtSubmission.py <path_to_npy_file> <memo>")
        sys.exit(1)

    npy_path = sys.argv[1]
    memo_text = sys.argv[2]

    creator = F01__CrtSubmission(path=npy_path, memo=memo_text)
    creator.f01__get_data()
    creator.f03__crt_submission()
    creator.f04__save_submission()