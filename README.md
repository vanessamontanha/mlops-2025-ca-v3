## Project Deployment Summary

This MLOps project predicts burnout risk based on synthetic workplace data. It includes model training, a Flask API for predictions and Continuous Integration/Deployment via GitHub Actions.

### Project Structure
- `ModelTraining/`: contains `train_model.py`, dataset and saved model.
- `ModelServing/`: contains `app.py` for serving the model via Flask.
- `monitor.py`: performs drift detection and retraining.
- `Dockerfile` and `requirements.txt`: define container build for deployment.
- `tests/`: includes unit tests for both training and API logic.
- `.github/workflows/`: automation scripts for CI/CD/CT/CM.

### Deployment Flow
WGit branching and CI/CD strategy:
1. Work on features in `feature/` branches.
2. Merge into `dev` after testing.
3. Final merge into `main` (prod).
4. From `main`, build and tag Docker image ( `burnout-api:vX`).
5. Push to Docker Hub and deploy on Google Cloud VM and Mini Kube
6. Continuous Monitoring via GitHub Actions triggers retraining if performance drops.
