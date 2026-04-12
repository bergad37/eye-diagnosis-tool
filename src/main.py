from pathlib import Path
from models.retfound import RETFoundLoader


def main():
    # Get the project root directory (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    weights_path = project_root / "artifacts" / "models" / "retfound.pth"
    
    loader = RETFoundLoader(
    repo_id="gadbertrand/retfound-eye-diagnosis",
    filename="retfound_cfp_vit_large_clean.pth",
    device="cpu"
)

    model = loader.load()
    print("RETFound loaded:", type(model))


if __name__ == "__main__":
    main()
