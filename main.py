import tensorflow as tf

def main():
    tf.config.list_physical_devices('GPU')
    print("Hello from lip-reading!")


if __name__ == "__main__":
    main()
