import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure credentials
FACE_KEY = os.getenv("FACE_API_KEY")
FACE_ENDPOINT = os.getenv("FACE_ENDPOINT_URL")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Initialize FaceClient
face_client = FaceClient(FACE_ENDPOINT, CognitiveServicesCredentials(FACE_KEY))

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_name = "infinionstaffimages"

def download_blob_image(blob_name):
    """Download image from Azure Blob Storage as a stream."""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        return BytesIO(blob_data)  # Return as a byte stream
    except Exception as e:
        print(f"Error downloading blob {blob_name}: {e}")
        return None

def detect_face_id(face_client, image_stream):
    """Detect face ID from an image stream."""
    detected_faces = face_client.face.detect_with_stream(
        image_stream,
        return_face_id=True
    )
    if detected_faces:
        return detected_faces[0].face_id
    else:
        raise ValueError(f"No face detected in the provided image.")

def verify_faces(face_client, face_id_1, face_id_2):
    """Verify if two faces belong to the same person."""
    try:
        verification_result = face_client.face.verify_face_to_face(
            face_id1=face_id_1,
            face_id2=face_id_2
        )
        return verification_result
    except Exception as e:
        print(f"Error verifying faces: {e}")
        return None

def main():
    # Blob names of the images to compare
    blob_name_1 = 'test-image/test_imagev3.jpg'
    blob_name_2 = 'test-image/test_imagev4.jpg'

    try:
        # Download the images from blob storage
        print("Downloading image 1 from blob storage...")
        image_stream_1 = download_blob_image(blob_name_1)
        print("Downloading image 2 from blob storage...")
        image_stream_2 = download_blob_image(blob_name_2)

        if image_stream_1 and image_stream_2:
            # Detect face IDs in both images
            print("Detecting face in image 1...")
            face_id_1 = detect_face_id(face_client, image_stream_1)

            print("Detecting face in image 2...")
            face_id_2 = detect_face_id(face_client, image_stream_2)

            if face_id_1 and face_id_2:
                # Verify if the two faces are the same
                print("Verifying faces...")
                result = verify_faces(face_client, face_id_1, face_id_2)
                
                if result:
                    if result.is_identical:
                        print(f"The faces match with a confidence of {result.confidence:.2f}")
                    else:
                        print("The faces do not match.")
                else:
                    print("Verification result is None.")
            else:
                print("Face detection failed for one or both images.")
        else:
            print("Failed to download one or both images from blob storage.")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
