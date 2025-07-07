# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project was developed by Juan Correa as part of the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project.
- **Model date**: July 2025
- **Model type**: Binary classification (RandomForestClassifier)
- **Framework**: Scikit-learn
- **Trained using**: One-hot encoded features, LabelBinarizer for label
- **GitHub repo**: https://github.com/juancl33/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
- **License**: Educational use
- **Contact**: [juancl33](https://github.com/juancl33)
## Intended Use
The model predicts whether a person earns more than $50,000 per year using U.S. Census data. It is designed for educational purposes only and is not intended for deployment in production settings.

- **Primary intended users**: Data science students and instructors
- **Out-of-scope uses**: Any real-world decision-making or hiring, loan, or financial applications
## Training Data
The model was trained on the UCI Census Income dataset (32,561 samples). Features include both continuous and categorical demographic attributes such as age, education, occupation, race, and sex.

- **Preprocessing**: Categorical features were one-hot encoded, and labels were binarized using LabelBinarizer.
## Evaluation Data
20% of the dataset was reserved for evaluation. Evaluation was performed on the test set and on categorical slices of the test data (e.g., by race, education, etc.).

- **Dataset**: UCI Census Income (held-out test split)
- **Preprocessing**: Same as training data
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Overall model performance on the test set:

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

Per-slice performance metrics are saved to `slice_output.txt`.
## Ethical Considerations
- The model reflects the biases present in the training data, which may include historical and societal inequalities.
- Sensitive features such as race, sex, and nationality are used, which could lead to biased outcomes.
## Caveats and Recommendations
- This model is trained for educational purposes and has not undergone fairness audits or extensive validation.
- Performance varies across demographic groups; interpret with caution.
- For production use, further tuning, fairness checks, and model monitoring would be necessary.