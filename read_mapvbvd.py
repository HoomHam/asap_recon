import json
from mapvbvd import mapVBVD
import argparse

def print_mapvbvd_header(twix_path: str) -> None:
    twix_obj = mapVBVD(twix_path)
    twix_list = twix_obj if isinstance(twix_obj, list) else [twix_obj]

    for idx, tw in enumerate(twix_list):
        print(f"\n--- Twix dataset {idx} ---")
        hdr = tw.hdr  # nested dict-like structure
        try:
            print(json.dumps(hdr, indent=2, default=str))
        except TypeError:
            # fallback: stringify non-serializable items
            print(json.dumps({k: str(v) for k, v in hdr.items()}, indent=2))

# Example:
# print_mapvbvd_header("meas_MID00000_FID00000.dat")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input MapVBVD file')
    args = parser.parse_args()
    print_mapvbvd_header(args.input)