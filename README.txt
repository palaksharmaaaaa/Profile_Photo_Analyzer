PROFILE PHOTO ANALYZER - AGE ESTIMATION UPDATE

What changed:
- InsightFace buffalo_l remains responsible for face detection, landmarks, and its gender-presentation prediction.
- The old InsightFace age attribute is no longer used.
- A separate FairFace image-classification model estimates an approximate age group.
- The output includes estimated_age_group, age_model, and age_model_score.

Install from the folder containing app.py:
  python -m pip install -r requirements.txt

Run the FastAPI app:
  python -m uvicorn app:app --reload

Then open:
  http://127.0.0.1:8000

First startup downloads pretrained model files, so internet access is needed initially. CPU inference may take some time.

Important:
- Age output is only an appearance-based estimate and can be wrong. Do not use it to verify someone's actual age or make consequential decisions.
- Model scores are classifier scores, not guaranteed calibrated probabilities.
- Review the licenses and usage terms of all pretrained models before commercial deployment.
