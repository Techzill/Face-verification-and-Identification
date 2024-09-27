import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
from PIL import Image  # Ensure PIL (Pillow) is installed

# Load environment variables from .env file
load_dotenv()

# Azure credentials loaded from environment variables
FACE_KEY = os.getenv("FACE_API_KEY")
FACE_ENDPOINT = os.getenv("FACE_ENDPOINT_URL")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not FACE_KEY or not FACE_ENDPOINT:
    raise EnvironmentError("Please set FACE_API_KEY and FACE_ENDPOINT_URL environment variables.")
if not AZURE_STORAGE_CONNECTION_STRING:
    raise EnvironmentError("Please set AZURE_STORAGE_CONNECTION_STRING environment variable.")

# Create a FaceClient instance
face_client = FaceClient(FACE_ENDPOINT, CognitiveServicesCredentials(FACE_KEY))

# Create BlobServiceClient using the connection string
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_name = ""

# Setup retry mechanism for network requests
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# Function to check if a person group exists
def check_person_group_exists(face_client, person_group_id):
    try:
        face_client.person_group.get(person_group_id)
        print(f"Person group '{person_group_id}' already exists.")
        return True
    except Exception as e:
        if "PersonGroupNotFound" in str(e):
            print(f"Person group '{person_group_id}' not found. A new one will be created.")
            return False
        else:
            raise e

# Function to create a dynamic person group
def create_person_group(face_client, person_group_id, person_group_name):
    try:
        print(f"Creating person group: {person_group_name}...")
        face_client.person_group.create(person_group_id=person_group_id, name=person_group_name)
        print(f"Person group '{person_group_name}' created.")
    except Exception as e:
        print(f"Error creating person group: {e}")

# Function to train the person group
def train_person_group(face_client, person_group_id):
    print("Training the person group...")
    face_client.person_group.train(person_group_id)
    while True:
        status = face_client.person_group.get_training_status(person_group_id)
        if status.status == 'succeeded':
            print("Person group trained successfully.")
            break
        elif status.status == 'failed':
            raise Exception("Training failed.")
        time.sleep(1)

# Function to add persons and their faces to the person group
def add_person_to_group(face_client, person_group_id, person_name, container_name):
    try:
        person = face_client.person_group_person.create(person_group_id, person_name)
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs()
        for blob in blobs:
            if blob.name.startswith(f'Infinion_images/{person_name}/') and blob.name.endswith('.jpg'):
                blob_client = container_client.get_blob_client(blob)
                try:
                    blob_data = blob_client.download_blob().readall()
                    with BytesIO(blob_data) as img_stream:
                        face_client.person_group_person.add_face_from_stream(person_group_id, person.person_id, img_stream)
                except Exception as e:
                    print(f"Error downloading or processing blob {blob.name}: {e}")
        print(f"Added faces for {person_name}.")
        return person.person_id
    except Exception as e:
        print(f"Error adding person '{person_name}' to group: {e}")

# Function to identify faces from the test image
def identify_faces(face_client, test_image_url, person_group_id):
    try:
        with session.get(test_image_url, stream=True) as response:
            response.raise_for_status()
            faces = face_client.face.detect_with_stream(response.raw)

        face_ids = [face.face_id for face in faces]
        if not face_ids:
            print("No faces detected.")
            return []

        results = face_client.face.identify(face_ids, person_group_id)
        identified_faces = []
        for face, result in zip(faces, results):
            if result.candidates:
                person = face_client.person_group_person.get(person_group_id, result.candidates[0].person_id)
                identified_faces.append({"bbox": face.face_rectangle, "personName": person.name})
            else:
                identified_faces.append({"bbox": face.face_rectangle, "personName": "Unknown"})

        return identified_faces
    except Exception as e:
        print(f"Error identifying faces: {e}")
        return []

# Function to plot the identified faces on the test image
def plot_faces_on_image(image_url, identified_faces):
    try:
        with session.get(image_url, stream=True) as response:
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Add a rectangle patch for each identified face
            for face in identified_faces:
                rect = patches.Rectangle(
                    (face['bbox'].left, face['bbox'].top),
                    face['bbox'].width, face['bbox'].height,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                # Add label for the identified person without background
                ax.text(face['bbox'].left, face['bbox'].top - 10, face['personName'], color='white')

            plt.show()
    except Exception as e:
        print(f"Error plotting faces on image: {e}")

def main():
    # Define URL for test image
    test_image_url = "https://infinionvisionstore.blob.core.windows.net/infinionstaffimages/test-image/test_imagev4.jpg"
    person_group_id = "infinion_person_group"

    # Check if the person group exists before creating
    if not check_person_group_exists(face_client, person_group_id):
        # Create a new person group if it doesn't exist
        create_person_group(face_client, person_group_id, "infinion_person_group")

        # Add persons to the group based on subfolder names
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs()
        person_names = set()
        for blob in blobs:
            if blob.name.startswith('Infinion_images/') and '/' in blob.name:
                person_name = blob.name.split('/')[1]
                person_names.add(person_name)

        for person_name in person_names:
            add_person_to_group(face_client, person_group_id, person_name, container_name)

        # Train the person group after adding persons
        train_person_group(face_client, person_group_id)

    # Identify faces in the test image
    identified_faces = identify_faces(face_client, test_image_url, person_group_id)

    if identified_faces:
        # Plot the identified faces on the test image
        plot_faces_on_image(test_image_url, identified_faces)
    else:
        print("No faces identified.")

if __name__ == "__main__":
    main()
