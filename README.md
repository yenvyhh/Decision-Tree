# Projekt 2 - Decision tree

**Bei Verwendung des Jupyter Notebooks und zur Durchführung der erforderten Übungsaufgaben, müssen zunächst folgende Installationen durchgeführt werden: 

**Um die Librarys in das Notebook zu importieren, müssen zu Beginn folgende Installationen einmalig durchgeführt werden (wenn für die vorherigen Übungen bereits getan, dann ignorieren):**
-> %conda install pandas 
-> %conda install numpy
-> %conda install sqlalchemy 
-> %conda install lxml
-> %conda install openpyxl 
-> %conda install xlrd 
-> %conda install matplotlib 
-> %conda install seaborn 
-> %conda install scikit-learn - sklearn
--> %conda install pydot
--> %conda install graphviz
--> %pip install pydot
--> %pip install graphviz
--> %pip install six

**Zu Beginn des Notebooks, werden die installierten Librarys wie folgt importiert:**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from IPython.display import Image

from six import StringIO
from sklearn.tree import export_graphviz
import pydot

from sklearn.ensemble import RandomForestClassifier


**Die Daten importieren,als DataFrame abspeichern und das Head anzeigen lassen:**
loans = pd.read_csv("Loan_Data.csv")
loans.head()
**Nach Ausführung sollte von der importierten Datei die ersten 5 Zeilen mit den Spalten ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'] angezeigt werden** 
       
**Informationen und Details des Data Frames bzw. der Daten anzeigen lassen:**     
loans.info()
loans.describe()
**Bei Info wird angezeigt, ob die Spalten einen Float, ein Integer oder ein Object sind. Zu dem wird bei RangeIndex angezeigt, dass es 9578 Einträge gibt. Bei Describe wird ein Dataset der Analyse geprintet. Beispiele hierfür sind der Durchschnittswert, der Minimum- oder Maximum-Wert.

**Darauffolgend erfolgt eine EXPLORATIVE DATENANALYSE, die durch verschiedene Diagrammvisualisierungen dargestellt werden. Ein Beispiel, das ausgeführt wird:**
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('fico')
**Durch Ausführen der ganzen Befehle werden zwei Histogramme (Balkendiagramme) in einer Abbildung erstellt. Dabei wird unterschieden in Credit.Policy = 1 und Credit Policy = 0. Beide Diagramme zeigen eine Verteilung des FICO Kreditscore des Kreditnehmers an. Dabei wird deutlich, dass das blaue Diagramm deutlich höher ist.


**Im nächsten Schritt werden die Daten vorbereitet und die KATEGORISCHEN EIGENSCHAFTEN, die als Datatype "Object" eingespeichert sind, umgewandelt. Nur so kann der Machine Learning Algorithmus damit arbeiten**
cat_feats = ['purpose']
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data
**Nach Ausführen von final_data wird ein neues DataFrame erstellt, wo die Spalte "purpose" gelöscht wurde. Rechts von dem DataFrame werden die neuen Spalten mit den kategorischen Eigenschaften von "Purpose" ergänzt, die mit 0 oder 1 zugewiesen sind.

**Die Daten werden nun in Trainings- und Test gesplittet. Dazu sollte zunächst definiert werden was das X-Array (Daten mit den Features) und was das y-Array (Daten mit der Zielvariable) ist:** 
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

**NachErstellung der Train- und Testdaten wird der Entscheidungsbaum trainiert und auf das Trainingsset gefittet:**
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

**Im Anschluss daran werden die Werte mit**
pred = dtree.predict(X_test)
**vorhergesagt. Basierend darauf kann ein Klassifizierungsreport und eine Confusion Matrix für das Modell erstellt werden:**
print (classification_report(y_test,pred))
print ("\n")
print(confusion_matrix(y_test,pred))
**Je näher die Werte bei precicion, recall und f1-score an 1 sind, desto genauer sind Auswertung. **

**Am Ende kann der Entscheidungsbaum noch mit den Werten des Random Forest verglichen werden. Dazu muss zunächst das Random Forest Modell trainiert werden:** 
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
**Nach Ausführen der obigen Befehle, ist in dem Klassifizierungs Report zusehen, dass die Werte besser sind als in dem vorherigen Report.
