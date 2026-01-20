from config.settings import IMAGE_SIZE
from data.preprocessing import get_preprocessing

def main():
    transform = get_preprocessing(IMAGE_SIZE)
    print("RETFound Eye Analysis project initialized.")

if __name__ == "__main__":
    main()
