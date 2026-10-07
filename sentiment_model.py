from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

reviews = [
	"I absolutely loved this movie",
	"This movie was terrible",
	"The acting was fantastic",
	"I hated every minute of it",
	"What a wonderful film",
	"This was boring and predictable"
]

labels = [
	"positive",
	"negative",
	"positive",
	"negative",
	"positive",
	"negative"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(reviews)
model = LogisticRegression()

model.fit(X, labels)

print(vectorizer.get_feature_names_out())
print(X.toarray())

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

print(model.classes_)
print("Intercept:", model.intercept_[0])

for word, weight in zip(feature_names, weights):
	print(f"{word:12} {weight:.3f}")

new_review = ["and this was absolutely terrible acting"]

new_X = vectorizer.transform(new_review)

prediction = model.predict(new_X)

probabilities = model.predict_proba(new_X)

print("Negative probability:", probabilities[0][0])
print("Positive probability:", probabilities[0][1])

print("Review:", new_review[0])
print("Prediction:", prediction[0])
