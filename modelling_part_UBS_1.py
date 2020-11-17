# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:34:02 2020

@author: 44563
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import confusion_matrix

import plotly.express as px

from PIL import Image,ImageFilter,ImageEnhance


from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import squarify

import copy

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from keras import models
from keras.utils import plot_model


from numpy.random import seed
seed(123)




import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Early Warning System Web App")
    st.sidebar.title("Early Warning System Web App")
    st.markdown("Binary Classification on Grocery Store Dataset  🏪")
    st.sidebar.markdown("Binary Classification on Grocery Store Dataset 🏪")

    @st.cache(persist=True)
    # def load_data():
    #     data=pd.read_csv('G://google download/Research/UBS_pitch/final_modelling_data_1.csv')
    #     #label = LabelEncoder()
    #     #for col in data.columns:
    #         #data[col] = label.fit_transform(data[col])
    #     return data
        
    @st.cache(persist=True)
    def split(df=None):
        #y = df.type
        #x = df.drop(columns = ['type'])
        #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
      #   labels = df['target']
      #   train_dataset, test_dataset = train_test_split(df, test_size=0.3, stratify=labels, random_state=3)
      #   train_features = train_dataset.copy()
      #   test_features = test_dataset.copy()

      #   train_labels = train_features.pop('target')
      #   test_labels = test_features.pop('target')
        
      #   train_features[['Sales_Change', 'Employee_Change', 'Civilian_labor_force_2019',
      #  'Median_Household_Income_2018', 'Location Employee Size Actual',
      #  'Location Sales Volume Actual', 'Years In Database', 'Square Footage',
      #  'Credit Score Alpha', 'Grocery_within_Zip',
      #  'Supermarket_within_5_miles', 'Google_Scores', 'Google_Reviews']] = normalizer.fit_transform(train_features[['Sales_Change', 'Employee_Change', 'Civilian_labor_force_2019',
      #  'Median_Household_Income_2018', 'Location Employee Size Actual',
      #  'Location Sales Volume Actual', 'Years In Database', 'Square Footage',
      #  'Credit Score Alpha', 'Grocery_within_Zip',
      #  'Supermarket_within_5_miles', 'Google_Scores', 'Google_Reviews']])
                                                                                                                    
      #   test_features[['Sales_Change', 'Employee_Change', 'Civilian_labor_force_2019',
      #  'Median_Household_Income_2018', 'Location Employee Size Actual',
      #  'Location Sales Volume Actual', 'Years In Database', 'Square Footage',
      #  'Credit Score Alpha', 'Grocery_within_Zip',
      #  'Supermarket_within_5_miles', 'Google_Scores', 'Google_Reviews']] = normalizer.transform(test_features[['Sales_Change', 'Employee_Change', 'Civilian_labor_force_2019',
      #  'Median_Household_Income_2018', 'Location Employee Size Actual',
      #  'Location Sales Volume Actual', 'Years In Database', 'Square Footage',
      #  'Credit Score Alpha', 'Grocery_within_Zip',
      #  'Supermarket_within_5_miles', 'Google_Scores', 'Google_Reviews']])
        x_train = pd.read_csv(os.path.join('final_train_features_3.csv'))
        y_train = pd.read_csv(os.path.join('final_train_labels_3.csv'))
        
        x_test = pd.read_csv(os.path.join('final_test_features_3.csv'))
        y_test = pd.read_csv(os.path.join('final_test_labels_3.csv'))                                                                                                          

        
        return x_train,x_test,y_train,y_test
    def plot_metrics(matrics_list):



        if 'Confusion Matrix' in matrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test_2,y_test, display_labels = class_names)
            st.pyplot()
        if 'ROC Curve' in matrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test_2,y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in matrics_list:
            st.subheader("Precision Recall")
            plot_precision_recall_curve(model,x_test_2,y_test)
            st.pyplot()
            
    def plot_metrics_2(matrics_list):

        if 'Confusion Matrix' in matrics_list:
            st.subheader("Confusion Matrix")
            #plot_confusion_matrix(model, x_test_2,y_test_2, display_labels = class_names)
            if_cm=confusion_matrix(y_test_2, y_pred)
            df_cm = pd.DataFrame(if_cm,
                  ['True Closed','True Open'],['Pred Closed','Pred Open'])
            plt.figure(figsize = (8,4))
            #sns.set(font_scale=1.4)#for label size
            st.write(sns.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g'))# font size
            st.pyplot()
            
    def plot_metrics_3(matrics_list):

        if 'Confusion Matrix' in matrics_list:
            st.subheader("Confusion Matrix")
            #plot_confusion_matrix(model, x_test_2,y_test_2, display_labels = class_names)
            if_cm=confusion_matrix(test_labels, test_NN_prediction)
            df_cm = pd.DataFrame(if_cm,
                  ['True Closed','True Open'],['Pred Closed','Pred Open'])
            plt.figure(figsize = (8,4))
            #sns.set(font_scale=1.4)#for label size
            st.write(sns.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g'))# font size
            st.pyplot()
        
    # Parameters Tuning By GridSearchCV
    def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy'):
        response = []
        gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score)
        # SearchOptimumParameters
        search = gridsearch.fit(train_x, train_y)
        # print("GridSearch's Optimum Parameter：", search.best_params_)
        # print("GridSearch's Optimum Score： %0.4lf" %search.best_score_)
        # predict_y = gridsearch.predict(test_x)
        # print("Accuracy: %0.4lf" %accuracy_score(test_y, predict_y))
        # response['predict_y'] = predict_y
        # response['accuracy_score'] = accuracy_score(test_y,predict_y)
        response.append(search.best_params_)
        response.append(search.best_score_)
        return response 
    
    @st.cache(persist=True)
    def explore_data_2(dataset=None):
        df_2 = pd.read_csv(os.path.join('closure_history_1.csv'))
        #df2 = pd.read_csv('G://google download/Research/UBS_pitch/closure_history_1.csv')
        return df_2 
    #df = load_data()
    
    def explore_data_3(dataset=None):
        df_3 = pd.read_csv(os.path.join('increased_stores_1.csv'))
        #df2 = pd.read_csv('G://google download/Research/UBS_pitch/closure_history_1.csv')
        return df_3 
    #df = load_data()
    
    x_train,x_test,y_train,y_test = split()
    df = pd.concat([y_train, x_train], axis=1)
    class_names = ['Closed','Open']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)","Logistic regression",
                                                    "Random Forest","XGBoost","Isolation Forest",
                                                    "Neural Network Ensemble"))
                                                    #,"Neural Network Ensemble"))
    
    if st.checkbox("Closure history of grocery stores in New Jersey"):
        data_2 = explore_data_2()
        #st.write(
        fig2 = px.scatter_mapbox(data_2, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        hover_data=['Status','Location Sales Volume Actual',
                                    'Location Employee Size Actual','Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma',,
                        color='Status', color_discrete_sequence=["green", "red"],
                         #size='PointSize',
                         animation_frame='Year'
                        )
        #fig2.update_layout(title_text = 'History of closed grocery stores in New Jersey',font_size=18)
        fig2.update_layout(mapbox_style="open-street-map")
        fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        #fig2.update_layout(margin=dict(l=20,r=0,b=0,t=70,pad=0),paper_bgcolor="white",font_size=18)
        st.plotly_chart(fig2)
        
    if st.checkbox("Insights from history 💡"):
        data_3 = explore_data_3()
        #st.write(
        fig3 = px.line(data_3, x="Year", y="Number of stores", title='Newly closed stores by year')
        #fig2.update_layout(title_text = 'History of closed grocery stores in New Jersey',font_size=18)
        
        st.plotly_chart(fig3)

# Try SVM
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Manual Variable Selection")
        selected_features = st.sidebar.multiselect("Which variables to choose?", ('Credit Score Alpha', 'Employee_Change', 'Estimated Labor Cost',
       'Google_Reviews', 'Google_Scores', 'Grocery_within_Zip', 'Sales_Change',
       'Sales_per_employee', 'Square Footage', 'Supermarket_within_5_miles',
       'county_personal_income', 'county_population'))
        
        if st.sidebar.button("Run Grid Search", key='Run Grid Search'):
            # BuildClassifiers
            x_train_1 = copy.deepcopy(x_train)
            x_train_1 = x_train_1[list(selected_features)]
            
            x_test_1 = copy.deepcopy(x_test)
            x_test_1 = x_test_1[list(selected_features)]
            
            classifiers = [SVC(probability=True)]
            classifier_names = ['SVMclassifier']

            # Parameters of Classifiers
            classifier_param_grid = [{'SVMclassifier__C':[1,5,9],
                                      'SVMclassifier__kernel': ('rbf','linear'),
                                      'SVMclassifier__gamma': ('scale','auto')
                                      }]
            # TuningFunctionCall
            for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
                pipeline = Pipeline([(model_name, model)])
                result = GridSearchCV_work(pipeline, x_train_1, y_train, x_test_1, y_test, model_param_grid , score = 'accuracy')
            #st.write("Accuracy: ", accuracy.round(4))
            st.write("GridSearch's Optimum Parameter：", result[0])
            st.write("GridSearch's Optimum Score：", result[1].round(4))
        
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C',value=9.0)
        kernel = st.sidebar.radio("Kernel",("rbf","linear"),key = 'kernel',index=0)
        gamma = st.sidebar.radio("Gamma (Kernel Cofficient)",("scale","auto"),key='gamma',index=0)

        metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results: ")
            model = SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
            
            x_train_2 = copy.deepcopy(x_train)
            x_train_2 = x_train_2[list(selected_features)]
            
            x_test_2 = copy.deepcopy(x_test)
            x_test_2 = x_test_2[list(selected_features)]

            model.fit(x_train_2,y_train)
            feature_names = list(x_train_2.columns)
            # importances = model.feature_importances_



            # df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
            # df_importance['importances'] = importances

            # df_importance = df_importance.sort_values(by='importances', ascending=False)
            # # Show Plots
            # st.markdown("Importance of selected features")
            #     #data = explore_data(my_dataset)
            # values_input = df_importance[df_importance['importances'] != 0]['importances']
            # fig = plt.gcf()
            # ax = fig.add_subplot()
            # fig.set_size_inches(10, 6)
            
            # st.write(squarify.plot(sizes=[round(i,3) for i in list(values_input)], 
            #   label=df_importance[df_importance['importances'] != 0]['feature_names'], alpha=.6 ,
            #   value=[round(i,3) for i in list(values_input)],color = ['green','white','red']),
            #          )
            #     # Use Matplotlib to render seaborn
            # st.pyplot()

            
            accuracy = model.score(x_test_2,y_test)
            y_pred = model.predict(x_test_2)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(y_test,y_pred,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(y_test,y_pred,labels=class_names).round(4))
            confusion_mat = confusion_matrix(y_test, y_pred)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))

            plot_metrics(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([x_train, y_train], axis=1), pd.concat([x_test, y_test], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = model.predict_proba(current_open_stores[list(selected_features)])[:, [0]]
            current_open_stores['risk_prob'] = risk_prob
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_prob'].apply(lambda x : convert_risk(x))
            
            current_open_stores = current_open_stores.sort_values(by=['risk_prob'],ascending=True).reset_index(drop=True)
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_prob',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_prob','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_prob'],ascending=False).reset_index(drop=True))

            
# Try Logistic Regression
    if classifier == 'Logistic regression':
        st.sidebar.subheader("Manual Variable Selection")
        selected_features = st.sidebar.multiselect("Which variables to choose?", ('Credit Score Alpha', 'Employee_Change', 'Estimated Labor Cost',
       'Google_Reviews', 'Google_Scores', 'Grocery_within_Zip', 'Sales_Change',
       'Sales_per_employee', 'Square Footage', 'Supermarket_within_5_miles',
       'county_personal_income', 'county_population'))
        
        if st.sidebar.button("Run Grid Search", key='Run Grid Search'):
            # BuildClassifiers
            x_train_1 = copy.deepcopy(x_train)
            x_train_1 = x_train_1[list(selected_features)]
            
            x_test_1 = copy.deepcopy(x_test)
            x_test_1 = x_test_1[list(selected_features)]
            
            classifiers = [LogisticRegression()]
            classifier_names = ['logisticregressionclassifier']

            # Parameters of Classifiers
            classifier_param_grid = [{'logisticregressionclassifier__C':[1,5,9],
                                      'logisticregressionclassifier__max_iter': [150,300,450]
                                      }]
            # TuningFunctionCall
            for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
                pipeline = Pipeline([(model_name, model)])
                result = GridSearchCV_work(pipeline, x_train_1, y_train, x_test_1, y_test, model_param_grid , score = 'accuracy')
            #st.write("Accuracy: ", accuracy.round(4))
            st.write("GridSearch's Optimum Parameter：", result[0])
            st.write("GridSearch's Optimum Score：", result[1].round(4))
            
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C_LR',value=1.0)
        max_iter = st.sidebar.slider("Maximum number of iterations", 100,500,key='max_iter',value=150)

        metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic regression Results: ")
            model = LogisticRegression(C=C,max_iter=max_iter)
            
            x_train_2 = copy.deepcopy(x_train)
            x_train_2 = x_train_2[list(selected_features)]
            
            x_test_2 = copy.deepcopy(x_test)
            x_test_2 = x_test_2[list(selected_features)]

            model.fit(x_train_2,y_train)
            feature_names = list(x_train_2.columns)
            # importances = model.feature_importances_



            # df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
            # df_importance['importances'] = importances

            # df_importance = df_importance.sort_values(by='importances', ascending=False)
            # # Show Plots
            # st.markdown("Importance of selected features")
            #     #data = explore_data(my_dataset)
            # values_input = df_importance[df_importance['importances'] != 0]['importances']
            # fig = plt.gcf()
            # ax = fig.add_subplot()
            # fig.set_size_inches(10, 6)
            
            # st.write(squarify.plot(sizes=[round(i,3) for i in list(values_input)], 
            #   label=df_importance[df_importance['importances'] != 0]['feature_names'], alpha=.6 ,
            #   value=[round(i,3) for i in list(values_input)],color = ['green','white','red']),
            #          )
            #     # Use Matplotlib to render seaborn
            # st.pyplot()

            
            accuracy = model.score(x_test_2,y_test)
            y_pred = model.predict(x_test_2)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(y_test,y_pred,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(y_test,y_pred,labels=class_names).round(4))
            
            confusion_mat = confusion_matrix(y_test, y_pred)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))
            
            plot_metrics(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([x_train, y_train], axis=1), pd.concat([x_test, y_test], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = model.predict_proba(current_open_stores[list(selected_features)])[:, [0]]
            current_open_stores['risk_prob'] = risk_prob
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_prob'].apply(lambda x : convert_risk(x))
            current_open_stores = current_open_stores.sort_values(by=['risk_prob'],ascending=True).reset_index(drop=True)
            
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_prob',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_prob','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_prob'],ascending=False).reset_index(drop=True))


# Try Random Forest
    if classifier == 'Random Forest':
        st.sidebar.subheader("Manual Variable Selection")
        selected_features = st.sidebar.multiselect("Which variables to choose?", ('Credit Score Alpha', 'Employee_Change', 'Estimated Labor Cost',
       'Google_Reviews', 'Google_Scores', 'Grocery_within_Zip', 'Sales_Change',
       'Sales_per_employee', 'Square Footage', 'Supermarket_within_5_miles',
       'county_personal_income', 'county_population'))
        
        if st.sidebar.button("Run Grid Search", key='Run Grid Search'):
            # BuildClassifiers
            x_train_1 = copy.deepcopy(x_train)
            x_train_1 = x_train_1[list(selected_features)]
            
            x_test_1 = copy.deepcopy(x_test)
            x_test_1 = x_test_1[list(selected_features)]
            
            classifiers = [RandomForestClassifier(random_state=3, criterion='gini')]
            classifier_names = ['randomforestclassifier']

            # Parameters of Classifiers
            classifier_param_grid = [{'randomforestclassifier__max_depth':[6,8,10],
                                      'randomforestclassifier__n_estimators': [800,1100,1200],
                                      'randomforestclassifier__bootstrap': (True,False)
                                      }]
            # TuningFunctionCall
            for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
                pipeline = Pipeline([(model_name, model)])
                result = GridSearchCV_work(pipeline, x_train_1, y_train, x_test_1, y_test, model_param_grid , score = 'accuracy')
            #st.write("Accuracy: ", accuracy.round(4))
            st.write("GridSearch's Optimum Parameter：", result[0])
            st.write("GridSearch's Optimum Score：", result[1].round(4))
            
        
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators=st.sidebar.number_input("The number of trees in the forest",100,5000,step = 10,key='n_estimator',value=1100)
        max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth",value=10)
        bootstrap = st.sidebar.radio("Bootstrap samples ",('True','False'),key='bootstrap',index=0)

        metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results: ")
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            x_train_2 = copy.deepcopy(x_train)
            x_train_2 = x_train_2[list(selected_features)]
            
            x_test_2 = copy.deepcopy(x_test)
            x_test_2 = x_test_2[list(selected_features)]

            model.fit(x_train_2,y_train)
            feature_names = list(x_train_2.columns)
            importances = model.feature_importances_



            df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
            df_importance['importances'] = importances

            df_importance = df_importance.sort_values(by='importances', ascending=False)
            # Show Plots
            st.markdown("Importance of selected features")
                #data = explore_data(my_dataset)
            #values_input = df_importance[df_importance['importances'] != 0]['importances']
            #fig = plt.gcf()
            #ax = fig.add_subplot()
            #fig.set_size_inches(10, 6)
            
            #st.write(squarify.plot(sizes=[round(i,3) for i in list(values_input)], 
              #label=df_importance[df_importance['importances'] != 0]['feature_names'], alpha=.6 ,
              #value=[round(i,3) for i in list(values_input)],color = ['green','white','red']),
                     #)
                # Use Matplotlib to render seaborn
            st.write(sns.barplot(y="feature_names", x="importances", data=df_importance))
            st.pyplot()

            
            accuracy = model.score(x_test_2,y_test)
            y_pred = model.predict(x_test_2)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(y_test,y_pred,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(y_test,y_pred,labels=class_names).round(4))
            
            confusion_mat = confusion_matrix(y_test, y_pred)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))
            
            plot_metrics(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([x_train, y_train], axis=1), pd.concat([x_test, y_test], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = model.predict_proba(current_open_stores[list(selected_features)])[:, [0]]
            current_open_stores['risk_prob'] = risk_prob
            
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_prob'].apply(lambda x : convert_risk(x))
            current_open_stores = current_open_stores.sort_values(by=['risk_prob'],ascending=True).reset_index(drop=True)
            
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_prob',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_prob','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_prob'],ascending=False).reset_index(drop=True))


# Try XGBoost 
    if classifier == 'XGBoost':
        st.sidebar.subheader("Manual Variable Selection")

        selected_features = st.sidebar.multiselect("Which variables to choose?", ('Credit Score Alpha', 'Employee_Change', 'Estimated Labor Cost',
       'Google_Reviews', 'Google_Scores', 'Grocery_within_Zip', 'Sales_Change',
       'Sales_per_employee', 'Square Footage', 'Supermarket_within_5_miles',
       'county_personal_income', 'county_population'))
        
        if st.sidebar.button("Run Grid Search", key='Run Grid Search'):
            # BuildClassifiers
            x_train_1 = copy.deepcopy(x_train)
            x_train_1 = x_train_1[list(selected_features)]
            
            x_test_1 = copy.deepcopy(x_test)
            x_test_1 = x_test_1[list(selected_features)]
            
           

            classifiers = [XGBClassifier(random_state=3, criterion='gini')]
            classifier_names = ['xgboostclassifier']

            # Parameters of Classifiers
            classifier_param_grid = [{'xgboostclassifier__max_depth':[6,8,10],
                                      'xgboostclassifier__n_estimators': [996,998,1000],
                                      'xgboostclassifier__min_child_weight': [4,5]
                                      }]
            # TuningFunctionCall
            for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
                pipeline = Pipeline([(model_name, model)])
                result = GridSearchCV_work(pipeline, x_train_1, y_train, x_test_1, y_test, model_param_grid , score = 'accuracy')
            #st.write("Accuracy: ", accuracy.round(4))
            st.write("GridSearch's Optimum Parameter：", result[0])
            st.write("GridSearch's Optimum Score：", result[1].round(4))
        
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators=st.sidebar.number_input("The number of gradient boosted trees",100,5000,step = 10,key='n_estimator',value=1000)
        max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth",value=6)
        min_child_weight = st.sidebar.number_input("Minimum sum of instance weight(hessian) needed in a child",1,15,step=1,
                                                   key='min_child_weight',value=2)


        
        metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        


        if st.sidebar.button("Classify", key='classify'):
            st.subheader("XGBoost Results: ")
            model = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,min_child_weight=min_child_weight)
            x_train_2 = copy.deepcopy(x_train)
            x_train_2 = x_train_2[list(selected_features)]
            
            x_test_2 = copy.deepcopy(x_test)
            x_test_2 = x_test_2[list(selected_features)]

            model.fit(x_train_2,y_train)
            feature_names = list(x_train_2.columns)
            importances = model.feature_importances_



            df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
            df_importance['importances'] = importances

            df_importance = df_importance.sort_values(by='importances', ascending=False)
            # Show Plots
            st.markdown("Importance of selected features")
                #data = explore_data(my_dataset)
            st.write(sns.barplot(y="feature_names", x="importances", data=df_importance))
                # Use Matplotlib to render seaborn
            st.pyplot()

            
            accuracy = model.score(x_test_2,y_test)
            y_pred = model.predict(x_test_2)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(y_test,y_pred,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(y_test,y_pred,labels=class_names).round(4))
            
            confusion_mat = confusion_matrix(y_test, y_pred)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))
            
            plot_metrics(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([x_train, y_train], axis=1), pd.concat([x_test, y_test], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = model.predict_proba(current_open_stores[list(selected_features)])[:, [0]]
            current_open_stores['risk_prob'] = risk_prob
            
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_prob'].apply(lambda x : convert_risk(x))
            
            current_open_stores = current_open_stores.sort_values(by=['risk_prob'],ascending=True).reset_index(drop=True)
            
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_prob',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            
            
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_prob','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_prob'],ascending=False).reset_index(drop=True))


# Try IsolationForest 
    if classifier == 'Isolation Forest':
        st.sidebar.subheader("Manual Variable Selection")

        selected_features = st.sidebar.multiselect("Which variables to choose?", ('Credit Score Alpha', 'Employee_Change', 'Estimated Labor Cost',
       'Google_Reviews', 'Google_Scores', 'Grocery_within_Zip', 'Sales_Change',
       'Sales_per_employee', 'Square Footage', 'Supermarket_within_5_miles',
       'county_personal_income', 'county_population'))
        
        if st.sidebar.button("Run Grid Search", key='Run Grid Search'):
            # BuildClassifiers
            x_train_1 = copy.deepcopy(x_train)
            x_train_1 = x_train_1[list(selected_features)]
            
            x_test_1 = copy.deepcopy(x_test)
            x_test_1 = x_test_1[list(selected_features)]
            

            y_train_1 = copy.deepcopy(y_train)
            y_train_1['target'] = y_train_1['target'].map(lambda x : 1 if x == 1 else -1)

            y_test_1 = copy.deepcopy(y_test)
            y_test_1['target'] = y_test_1['target'].map(lambda x : 1 if x == 1 else -1)
           

            classifiers = [IsolationForest(random_state=3)]
            classifier_names = ['isolationforestclassifier']

            # Parameters of Classifiers
            classifier_param_grid = [{'isolationforestclassifier__n_jobs':[-1,1,2],
                                      'isolationforestclassifier__n_estimators': [996,998,1000],
                                      'isolationforestclassifier__bootstrap': (True,False)
                                      }]
            # TuningFunctionCall
            for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
                pipeline = Pipeline([(model_name, model)])
                result = GridSearchCV_work(pipeline, x_train_1, y_train_1, x_test_1, y_test_1, model_param_grid , score = 'accuracy')
            #st.write("Accuracy: ", accuracy.round(4))
            st.write("GridSearch's Optimum Parameter：", result[0])
            st.write("GridSearch's Optimum Score：", result[1].round(4))
        
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators=st.sidebar.number_input("The number of base estimators in the ensemble",100,5000,step = 10,key='n_estimator',value=996)
        n_jobs = st.sidebar.number_input("The number of jobs to run in parallel for both fit and predict",-1,20,step=1,key="n_job",value=-1)
        bootstrap = st.sidebar.radio("Bootstrap samples ",('True','False'),key='bootstrap',index=1)


        
        metrics = st.sidebar.multiselect("What metrices to plot?", ['Confusion Matrix'])
        


        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Isolation Forest Results: ")
            model = IsolationForest(n_estimators=n_estimators,n_jobs=n_jobs,bootstrap=bootstrap)
            x_train_2 = copy.deepcopy(x_train)
            x_train_2 = x_train_2[list(selected_features)]
            
            x_test_2 = copy.deepcopy(x_test)
            x_test_2 = x_test_2[list(selected_features)]

            y_train_2 = copy.deepcopy(y_train)
            y_train_2['target'] = y_train_2['target'].map(lambda x : 1 if x == 1 else -1)

            y_test_2 = copy.deepcopy(y_test)
            y_test_2['target'] = y_test_2['target'].map(lambda x : 1 if x == 1 else -1)

            model.fit(x_train_2,y_train_2)
            feature_names = list(x_train_2.columns)
            # importances = model.feature_importances_



            # df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
            # df_importance['importances'] = importances

            # df_importance = df_importance.sort_values(by='importances', ascending=False)
            # # Show Plots
            # st.markdown("Importance of selected features")
            #     #data = explore_data(my_dataset)
            # values_input = df_importance[df_importance['importances'] != 0]['importances']
            # fig = plt.gcf()
            # ax = fig.add_subplot()
            # fig.set_size_inches(10, 6)
            
            # st.write(squarify.plot(sizes=[round(i,3) for i in list(values_input)], 
            #   label=df_importance[df_importance['importances'] != 0]['feature_names'], alpha=.6 ,
            #   value=[round(i,3) for i in list(values_input)],color = ['green','white','red']),
            #          )
            #     # Use Matplotlib to render seaborn
            # st.pyplot()

            
            #accuracy = model.score(x_test_2,y_test)
            y_pred = model.predict(x_test_2)
            #y_pred = [x if x == 1 else 0 for x in list(y_pred)]
            accuracy = accuracy_score(y_test_2, y_pred)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(y_test_2,y_pred,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(y_test_2,y_pred,labels=class_names).round(4))
            
            confusion_mat = confusion_matrix(y_test_2, y_pred)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))
            
            plot_metrics_2(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([x_train, y_train], axis=1), pd.concat([x_test, y_test], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = model.score_samples(current_open_stores[list(selected_features)])
            current_open_stores['risk_factor'] = abs(risk_prob)
            
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_factor'].apply(lambda x : convert_risk(x))
            current_open_stores = current_open_stores.sort_values(by=['risk_factor'],ascending=True).reset_index(drop=True)
            
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_factor',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            

            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_factor','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_factor'],ascending=False).reset_index(drop=True))



# Try Neural Network Ensemble 
    if classifier == 'Neural Network Ensemble':
        
        st.sidebar.subheader("Best Models Selection")

        selected_features = st.sidebar.multiselect("Which models to choose?", ('XGBoost_8',
             'XGBoost_7','XGBoost_6','XGBoost_9','RandomForest_6','RandomForest_2',
             'RandomForest_4','RandomForest_5','RandomForest_3'))
        
# =============================================================================
#         if st.sidebar.button("Run Grid Search", key='Run Grid Search'):
#             # BuildClassifiers
#             x_train_1 = copy.deepcopy(x_train)
#             x_train_1 = x_train_1[list(selected_features)]
#             
#             x_test_1 = copy.deepcopy(x_test)
#             x_test_1 = x_test_1[list(selected_features)]
#             
# 
#             y_train_1 = copy.deepcopy(y_train)
#             y_train_1['target'] = y_train_1['target'].map(lambda x : 1 if x == 1 else -1)
# 
#             y_test_1 = copy.deepcopy(y_test)
#             y_test_1['target'] = y_test_1['target'].map(lambda x : 1 if x == 1 else -1)
#            
# 
#             classifiers = [IsolationForest(random_state=3)]
#             classifier_names = ['isolationforestclassifier']
# 
#             # Parameters of Classifiers
#             classifier_param_grid = [{'isolationforestclassifier__n_jobs':[-1,1,2],
#                                       'isolationforestclassifier__n_estimators': [996,998,1000],
#                                       'isolationforestclassifier__bootstrap': (True,False)
#                                       }]
#             # TuningFunctionCall
#             for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
#                 pipeline = Pipeline([(model_name, model)])
#                 result = GridSearchCV_work(pipeline, x_train_1, y_train_1, x_test_1, y_test_1, model_param_grid , score = 'accuracy')
#             #st.write("Accuracy: ", accuracy.round(4))
#             st.write("GridSearch's Optimum Parameter：", result[0])
#             st.write("GridSearch's Optimum Score：", result[1].round(4))
# =============================================================================
        
        st.sidebar.subheader("Model Hyperparameters")
        optimizer = st.sidebar.radio("Optimizers",['Adam','SGD','RMSprop','Adagrad'],key='optimizer',index=0)
        epochs = st.sidebar.number_input("Number of Epochs",10,100,step = 10,key='epochs',value=50)
        batch_size = st.sidebar.number_input("Batch Size",40,140,step = 20,key='batch_size',value=100)
        # n_estimators=st.sidebar.number_input("The number of base estimators in the ensemble",100,5000,step = 10,key='n_estimator',value=996)
        # n_jobs = st.sidebar.number_input("The number of jobs to run in parallel for both fit and predict",-1,20,step=1,key="n_job",value=-1)
        # bootstrap = st.sidebar.radio("Bootstrap samples ",('True','False'),key='bootstrap',index=1)
        prob_threshold = st.sidebar.number_input("Probability Threshold",0.2,0.7,step = 0.1,key='prob_threshold',value=0.5)

          


        
        metrics = st.sidebar.multiselect("What metrices to plot?", ['Confusion Matrix'])
        


        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Neural Network Ensemble Results: ")
            train_predictions = pd.read_csv(os.path.join('final_final_train_predictions_2.csv'))
            train_predictions_2 = copy.deepcopy(train_predictions)
            train_predictions_2 = train_predictions_2[list(selected_features)]
            
            test_predictions = pd.read_csv(os.path.join('final_final_test_predictions_2.csv'))
            test_predictions_2 = copy.deepcopy(test_predictions)
            test_predictions_2 = test_predictions_2[list(selected_features)]
            
            train_labels = train_predictions['target']
            test_labels = test_predictions['target']
            
            X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(train_predictions_2, train_labels, test_size=0.3,stratify=train_labels, random_state=3) 
            model = Sequential()

            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            
            history = model.fit(X_final_train, y_final_train, epochs=epochs, validation_data=(X_final_val, y_final_val), batch_size=batch_size)
            st.subheader('Learning Curve: ')
            epochs = range(epochs)
            loss = history.history['loss']
            val_loss = history.history['val_loss']
                
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            
            df_loss = pd.DataFrame(data=loss, columns=["loss"])
            df_loss['cat'] = 'training_loss'
            df_loss['epochs'] = epochs
            
            df_val_loss = pd.DataFrame(data=val_loss, columns=["loss"])
            df_val_loss['cat'] = 'validation_loss'
            df_val_loss['epochs'] = epochs
            
            df_merge_loss = pd.concat([df_loss, df_val_loss])
            
            ### 
            df_acc = pd.DataFrame(data=acc, columns=["accuracy"])
            df_acc['cat'] = 'training_accuracy'
            df_acc['epochs'] = epochs
            
            df_val_acc = pd.DataFrame(data=val_acc, columns=["accuracy"])
            df_val_acc['cat'] = 'validation_accuracy'
            df_val_acc['epochs'] = epochs
            
            df_merge_acc = pd.concat([df_acc, df_val_acc])
            
            st.write(sns.lineplot(data=df_merge_loss, x="epochs", y="loss", hue="cat"))
            # Use Matplotlib to render seaborn
            st.pyplot()
            
            st.write(sns.lineplot(data=df_merge_acc, x="epochs", y="accuracy", hue="cat"))
            # Use Matplotlib to render seaborn
            st.pyplot()
            
            
            # plt.title('Training and Validation loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # #plt.legend()    
            # st.write(plt.plot(epochs, loss, label='loss'))
            # st.write(plt.plot(epochs, val_loss, label='val_loss'))
            
            # #plt.show()
            # plt.title('Training and Validation accuarcy')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # #plt.legend()    
            # st.write(plt.plot(epochs, acc, label='accuracy'))
            # st.write(plt.plot(epochs, val_acc, label='val_accuracy'))
            
            # st.subheader('Model parameters: ')
            # st.write(model.summary())
            #st.write（plot_model(model)）
            
            test_NN_prediction_prob = model.predict(test_predictions_2)
            test_NN_prediction =np.where(test_NN_prediction_prob > prob_threshold, 1, 0)
            
            #plt.show()
            
# =============================================================================
#             model = IsolationForest(n_estimators=n_estimators,n_jobs=n_jobs,bootstrap=bootstrap)
#             x_train_2 = copy.deepcopy(x_train)
#             x_train_2 = x_train_2[list(selected_features)]
#             
#             x_test_2 = copy.deepcopy(x_test)
#             x_test_2 = x_test_2[list(selected_features)]
# 
#             y_train_2 = copy.deepcopy(y_train)
#             y_train_2['target'] = y_train_2['target'].map(lambda x : 1 if x == 1 else -1)
# 
#             y_test_2 = copy.deepcopy(y_test)
#             y_test_2['target'] = y_test_2['target'].map(lambda x : 1 if x == 1 else -1)
# 
#             model.fit(x_train_2,y_train_2)
#             feature_names = list(x_train_2.columns)
#             # importances = model.feature_importances_
# 
# 
# 
#             # df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
#             # df_importance['importances'] = importances
# 
#             # df_importance = df_importance.sort_values(by='importances', ascending=False)
#             # # Show Plots
#             # st.markdown("Importance of selected features")
#             #     #data = explore_data(my_dataset)
#             # values_input = df_importance[df_importance['importances'] != 0]['importances']
#             # fig = plt.gcf()
#             # ax = fig.add_subplot()
#             # fig.set_size_inches(10, 6)
#             
#             # st.write(squarify.plot(sizes=[round(i,3) for i in list(values_input)], 
#             #   label=df_importance[df_importance['importances'] != 0]['feature_names'], alpha=.6 ,
#             #   value=[round(i,3) for i in list(values_input)],color = ['green','white','red']),
#             #          )
#             #     # Use Matplotlib to render seaborn
#             # st.pyplot()
# 
# =============================================================================
            
            accuracy = accuracy_score(test_labels, test_NN_prediction)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(test_labels,test_NN_prediction,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(test_labels,test_NN_prediction,labels=class_names).round(4))
            
            confusion_mat = confusion_matrix(test_labels, test_NN_prediction)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))
            
            plot_metrics_3(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([train_predictions], axis=1), pd.concat([test_predictions], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = 1 - model.predict(current_open_stores[list(selected_features)])
            current_open_stores['risk_prob'] = risk_prob
            
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_prob'].apply(lambda x : convert_risk(x))
            
            current_open_stores = current_open_stores.sort_values(by=['risk_prob'],ascending=True).reset_index(drop=True)
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_prob',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            

            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_prob','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_prob'],ascending=False).reset_index(drop=True))


        if st.sidebar.button("Classify with all models", key='classify_2'):
            st.subheader("Neural Network Ensemble Results: ")
            train_predictions = pd.read_csv(os.path.join('final_final_train_predictions_2.csv'))
            train_predictions_3 = copy.deepcopy(train_predictions)
            cols = [col for col in train_predictions_3.columns if col not in ['Company Name', 
                                                'lat', 'long', 'NeuralNetworkEnsemble', 'NeuralNetworkEnsemble_risk_prob']]
            #df2 = df[cols]
            train_predictions_3 = train_predictions_3[cols]
            
            test_predictions = pd.read_csv(os.path.join('final_final_test_predictions_2.csv'))
            test_predictions_3 = copy.deepcopy(test_predictions)
            cols_2 = [col for col in test_predictions_3.columns if col not in ['Company Name', 
                              'lat', 'long', 'NeuralNetworkEnsemble', 
                              'NeuralNetworkEnsemble_risk_prob']]
            test_predictions_3 = test_predictions_3[cols_2]
            
            train_labels = train_predictions_3['target']
            test_labels = test_predictions_3['target']
            
            X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(train_predictions_3, train_labels, test_size=0.3,stratify=train_labels, random_state=3) 
            model = Sequential()

            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            
            history = model.fit(X_final_train, y_final_train, epochs=epochs, validation_data=(X_final_val, y_final_val), batch_size=batch_size)
            st.subheader('Learning Curve: ')
            epochs = range(epochs)
            loss = history.history['loss']
            val_loss = history.history['val_loss']
                
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            
            df_loss = pd.DataFrame(data=loss, columns=["loss"])
            df_loss['cat'] = 'training_loss'
            df_loss['epochs'] = epochs
            
            df_val_loss = pd.DataFrame(data=val_loss, columns=["loss"])
            df_val_loss['cat'] = 'validation_loss'
            df_val_loss['epochs'] = epochs
            
            df_merge_loss = pd.concat([df_loss, df_val_loss])
            
            ### 
            df_acc = pd.DataFrame(data=acc, columns=["accuracy"])
            df_acc['cat'] = 'training_accuracy'
            df_acc['epochs'] = epochs
            
            df_val_acc = pd.DataFrame(data=val_acc, columns=["accuracy"])
            df_val_acc['cat'] = 'validation_accuracy'
            df_val_acc['epochs'] = epochs
            
            df_merge_acc = pd.concat([df_acc, df_val_acc])
            
            st.write(sns.lineplot(data=df_merge_loss, x="epochs", y="loss", hue="cat"))
            # Use Matplotlib to render seaborn
            st.pyplot()
            
            st.write(sns.lineplot(data=df_merge_acc, x="epochs", y="accuracy", hue="cat"))
            # Use Matplotlib to render seaborn
            st.pyplot()
            
            
            # plt.title('Training and Validation loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # #plt.legend()    
            # st.write(plt.plot(epochs, loss, label='loss'))
            # st.write(plt.plot(epochs, val_loss, label='val_loss'))
            
            # #plt.show()
            # plt.title('Training and Validation accuarcy')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # #plt.legend()    
            # st.write(plt.plot(epochs, acc, label='accuracy'))
            # st.write(plt.plot(epochs, val_acc, label='val_accuracy'))
            
            # st.subheader('Model parameters: ')
            # st.write(model.summary())
            #st.write（plot_model(model)）
            
            test_NN_prediction_prob = model.predict(test_predictions_3)
            test_NN_prediction =np.where(test_NN_prediction_prob > prob_threshold, 1, 0)
            
            #plt.show()
            
# =============================================================================
#             model = IsolationForest(n_estimators=n_estimators,n_jobs=n_jobs,bootstrap=bootstrap)
#             x_train_2 = copy.deepcopy(x_train)
#             x_train_2 = x_train_2[list(selected_features)]
#             
#             x_test_2 = copy.deepcopy(x_test)
#             x_test_2 = x_test_2[list(selected_features)]
# 
#             y_train_2 = copy.deepcopy(y_train)
#             y_train_2['target'] = y_train_2['target'].map(lambda x : 1 if x == 1 else -1)
# 
#             y_test_2 = copy.deepcopy(y_test)
#             y_test_2['target'] = y_test_2['target'].map(lambda x : 1 if x == 1 else -1)
# 
#             model.fit(x_train_2,y_train_2)
#             feature_names = list(x_train_2.columns)
#             # importances = model.feature_importances_
# 
# 
# 
#             # df_importance = pd.DataFrame(feature_names, columns=['feature_names']) 
#             # df_importance['importances'] = importances
# 
#             # df_importance = df_importance.sort_values(by='importances', ascending=False)
#             # # Show Plots
#             # st.markdown("Importance of selected features")
#             #     #data = explore_data(my_dataset)
#             # values_input = df_importance[df_importance['importances'] != 0]['importances']
#             # fig = plt.gcf()
#             # ax = fig.add_subplot()
#             # fig.set_size_inches(10, 6)
#             
#             # st.write(squarify.plot(sizes=[round(i,3) for i in list(values_input)], 
#             #   label=df_importance[df_importance['importances'] != 0]['feature_names'], alpha=.6 ,
#             #   value=[round(i,3) for i in list(values_input)],color = ['green','white','red']),
#             #          )
#             #     # Use Matplotlib to render seaborn
#             # st.pyplot()
# 
# =============================================================================
            
            accuracy = accuracy_score(test_labels, test_NN_prediction)
            st.write("Test Accuracy: ", accuracy.round(4))
            st.write("Test Precission: ",precision_score(test_labels,test_NN_prediction,labels=class_names).round(4))
            st.write("Test Recall: ",recall_score(test_labels,test_NN_prediction,labels=class_names).round(4))
            
            confusion_mat = confusion_matrix(test_labels, test_NN_prediction)
            spec = confusion_mat[0, 0]/(confusion_mat[0, 0]+confusion_mat[0, 1])
            st.write("Test Specificity: ", spec.round(4))
            
            plot_metrics_3(metrics)

            st.subheader("Prediction for the closure risk of different stores in 2020: ")
            #if st.checkbox("Predicted Stores at different levels of risk of closure"):
            all_dataset = pd.concat([pd.concat([train_predictions], axis=1), pd.concat([test_predictions], axis=1)])
            current_open_stores = all_dataset[all_dataset['target'] == 1].dropna()
                
            risk_prob = 1 - model.predict(current_open_stores[cols_2])
            current_open_stores['risk_prob'] = risk_prob
            
            def convert_risk(risk):
                if risk <= 0.4:
                    risk = 'low_risk'
                elif 0.4 < risk <= 0.6:
                    risk = 'medium_risk'
                else:
                    risk = 'high_risk'
                return risk
            current_open_stores['risk_level'] = current_open_stores['risk_prob'].apply(lambda x : convert_risk(x))
            
            current_open_stores = current_open_stores.sort_values(by=['risk_prob'],ascending=True).reset_index(drop=True)
            fig = px.scatter_mapbox(current_open_stores, lat="lat", lon="long", zoom=7, height=600,hover_name='Company Name',
                        #hover_data=['Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma', 
                        size='risk_prob',
                        color='risk_level', color_discrete_sequence=["green","red", "purple"])
            

            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
            
            st.subheader("Predicted list of Grocercy Stores at medium and high risk of closure: ")
            risky_stores = current_open_stores[current_open_stores['risk_level'] != 'low_risk'][['Company Name','risk_level','risk_prob','lat','long']]
            st.dataframe(risky_stores.sort_values(by=['risk_prob'],ascending=False).reset_index(drop=True))



    if st.sidebar.checkbox("Show input training data ",False):
        st.subheader("Grocery Store Data Set (Classification)")
        st.write(x_train)

        
if __name__ == '__main__':
    main()
