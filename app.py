import os
import streamlit as st
import cv2
import tempfile
from clarifai.grpc import service_pb2_grpc, service_pb2, resources_pb2
from clarifai.grpc.grpc.api.status import status_code_pb2
from clarifai.grpc.channel.clarifai_channel import ClarifaiChannel

st.title('Apparel Detection in Video using Clarifai')

CLARIFAI_API_KEY = os.environ.get('CLARIFAI_API_KEY')

if not CLARIFAI_API_KEY:
    st.error("Please set the CLARIFAI_API_KEY environment variable.")
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Open the video file with OpenCV
        cap = cv2.VideoCapture(tfile.name)
        
        # Extract frames at 1 frame per second
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while cap.isOpened():
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % int(frame_rate) == 0:
                frames.append(frame)
        cap.release()
        
        st.write(f"Extracted {len(frames)} frames from the video.")
        
        # Initialize Clarifai gRPC channel and stub
        stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
        metadata = (('authorization', f'Key {CLARIFAI_API_KEY}'),)
        
        # Apparel model ID
        model_id = 'e0be3b9d6a454f0493ac3a30784001ff'
        
        for idx, frame in enumerate(frames):
            # Encode frame as JPEG
            retval, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()
            
            # Create Clarifai image
            image = resources_pb2.Image(base64=img_bytes)
            
            # Build the request
            request = service_pb2.PostModelOutputsRequest(
                model_id=model_id,
                inputs=[
                    resources_pb2.Input(data=resources_pb2.Data(image=image))
                ]
            )
            
            # Get response
            response = stub.PostModelOutputs(request, metadata=metadata)
            
            if response.status.code != status_code_pb2.SUCCESS:
                st.write(f"Request failed for frame {idx}, status code: {response.status.code}")
                continue
            
            # Get outputs
            concepts = response.outputs[0].data.concepts
            
            # Display frame and predictions
            st.image(frame, channels="BGR", caption=f"Frame {idx}")
            st.write("Predicted apparel items:")
            for concept in concepts:
                st.write(f"{concept.name} ({concept.value * 100:.2f}%)")
