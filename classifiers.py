from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def classifiers(X_train_tf, X_test_tf, Y_train, Y_test):
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
    }

    # for name, clf in classifiers.items():
    #     clf.fit(X_train_tf, Y_train)
    #     Y_pred_tf = clf.predict(X_test_tf)
    #     acc = accuracy_score(Y_test, Y_pred_tf)
    #     print(f"\n\nClassification Report for {name}")
    #     print(f"Accuracy: {acc * 100}%")
    #     print(classification_report(Y_test, Y_pred_tf))

    lg = LogisticRegression(max_iter=1000)
    lg.fit(X_train_tf, Y_train)
    lg_y_predict = lg.predict(X_test_tf)

    svc = SVC(max_iter=1000)
    svc.fit(X_train_tf, Y_train)
    svc_y_predict = svc.predict(X_test_tf)

    return lg, lg_y_predict, svc, svc_y_predict
