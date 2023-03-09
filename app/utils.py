import pickle
import json
import numpy as np

class prediction():
    def __init__(self):
        pass
        
    def load_raw(self):

        with open("artifacts\insurance.pkl","rb") as model_file:
            self.model=pickle.load(model_file)
        with open('artifacts\columns_names.json','r') as col_file: 
            self.column_names = json.load(col_file)
        with open('artifacts\encoded_data.json','r') as encode_file:
            self.encoded_data = json.load(encode_file)

    def predict_cover(self,data):

        self.load_raw()
        self.data= data   
        
        user_input = np.zeros(len(self.column_names['Column Names']))

        array = np.array(self.column_names['Column Names'])

        age=data["html_age"]
        sex=data["html_sex"]
        bmi=data["html_bmi"]
        children=data["html_childs"]
        smoker=data["html_smoke"]
        region="southeast"

        user_input[0]=age         
        user_input[1]=self.encoded_data['sex'][sex]
        user_input[2]=bmi                       
        user_input[3]=children
        user_input[4]=self.encoded_data['smoker'][smoker]

        region_string = 'region_'+region
        region_index = np.where(array == region_string)[0][0]
        user_input[region_index] = 1 
        
        print(f"{user_input=}")
        expenses = self.model.predict([user_input])
        return (f"Predicted Expenses = {expenses}")
    
if __name__ == "__main__":

    pred_obj=prediction()
    pred_obj.load_raw