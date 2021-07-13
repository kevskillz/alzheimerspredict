from matplotlib.pyplot import plot


class PredictAzlheimersData:
    
    def __init__(self):
        import pickle
        with open(f'alzheimersDemographic.model', 'rb') as file:
            self.model = pickle.load(file)
        self.refDict = {0: "No Alzheimer's", 2: "Mild Alzheimer's", 1: "Very Mild Alzheimer's", 3: "Moderate Alzheimer's"}
    
    def predict(self, Gender, Age, EDUC, ASF, eTIV):
        if Gender == 'M':
            Gender = 1
        else: 
            Gender = 0
        

        return self.refDict[self.model.predict([[Gender, Age, EDUC, ASF, eTIV]])[0]]
   

class PredictAzlheimersImg: 
    def __init__(self):
        from keras.models import load_model
       
        self.model = load_model('alzheimersImg.model')
        self.refDict = {0: "Mild Alzheimer's", 1: "Moderate Alzheimer's",  2: "No Alzheimer's", 3: "Very Mild Alzheimer's"}
        self.dimensions = (180, 180)

   
       
    def predict(self, t_path):
        from keras_preprocessing import image
        import numpy as np
        img = image.load_img(t_path, target_size=self.dimensions, interpolation='bilinear')
        imgArr = image.img_to_array(img) / 255
        imgArr = np.array([imgArr])
        

        prediction = np.array(self.model.predict(imgArr)).ravel()
    
        resArr = {}
        for i, value in enumerate(prediction):
            
            resArr[self.refDict[i]] = value
            
        
        return resArr
    
    def show_model(self):
        from keras.utils.vis_utils import plot_model
        plot_model(self.model)

    


modelD = PredictAzlheimersData()
print(modelD.predict('M', 68, 18, 1.0, 1449)) # ~95.7575% accuracy
# normal person, should be No Alzheimer's

print(modelD.predict('F', 75, 12, 1.293, 1357))
# Mild Alzheimer's Patient

modelI = PredictAzlheimersImg() # ~94.14% accuracy

print(modelI.predict('images\\MildDemented\\28.jpg'))
# should be Mild Alzheimer's


