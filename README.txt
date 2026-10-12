# Profile Photo Analyzer

## Overview

Profile Photo Analyzer is a computer vision-based application designed to analyze uploaded profile photos. It checks facial presence, face count, face visibility, estimated age group, and image type to help determine whether an image meets profile photo requirements.

## Features

- **Face Detection:** Detects faces in uploaded profile photos.
- **Multiple Face Detection:** Identifies images containing multiple faces.
- **Face Visibility Analysis:** Checks whether a face is sufficiently visible for profile use.
- **Age Estimation:** Estimates the age group of a detected face.
- **Gender Presentation Estimation:** Provides a model-based gender presentation estimate.
- **Image Type Screening:** Helps identify real photographs, cartoons, anime, illustrations, and other image types.
- **Visual Results:** Displays the uploaded image and annotated face detection results.
- **Profile Photo Assessment:** Presents analysis results and warnings in a user-friendly interface.

## Technologies Used

- Python
- FastAPI
- OpenCV
- InsightFace
- Hugging Face Transformers
- NumPy
- Pillow
- HTML and CSS

## Project Structure

```text
profile-photo-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
├── uploads/
└── templates/
```

*The structure above is illustrative. Adjust it to match your actual project files.*

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/profile-photo-analyzer.git
cd profile-photo-analyzer
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository name.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the dependencies used by your project.

### 4. Run the Application

```bash
uvicorn app:app --reload
```

Open the following URL in your browser:

http://127.0.0.1:8000

## How It Works

1. Upload a profile photo.
2. The application processes the image.
3. The face detection model identifies visible faces.
4. The application estimates the age group and gender presentation when supported by the detected face.
5. Image type screening evaluates whether the image resembles a photograph or an illustration.
6. The interface displays the analysis results, warnings, and annotated image.

## Important Notes

- Age and gender presentation are model estimates and may be inaccurate.
- Image type screening is probabilistic and can misclassify photographs and illustrations.
- A face detection result does not guarantee that an image is authentic or unaltered.
- Do not use estimated age or gender as definitive proof of a person's identity or age.
- Avoid uploading sensitive personal images without appropriate authorization.

## Future Improvements

- Improve real-photo and AI-generated-image classification.
- Enhance face completeness and image-quality checks.
- Improve age-group estimation accuracy.
- Add confidence scores and clearer error messages.
- Add automated testing and model evaluation.

## License

Add a license file if you intend to distribute this project publicly.
