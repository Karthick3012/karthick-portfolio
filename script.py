import webbrowser
import os

# Open the portfolio HTML in the default browser
file_path = os.path.abspath("index.html")
webbrowser.open(f"file://{file_path}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
df = pd.read_csv("Telco-Customer-Churn.csv")

sns.countplot(x='tenure', hue='Churn', data=df)
plt.title('Churn Rate by Tenure')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/churn_chart.png')

import plotly.express as px
df = px.data.gapminder().query("year==2007")

fig = px.scatter(df, x="gdpPercap", y="lifeExp", color="continent", size="pop")
fig.write_html("visualizations/dashboard.html")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Your model
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

plt.savefig("models/churn_model_output.png")
