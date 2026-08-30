import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score


class ApartmentValModel:


    def __init__(self):

        self.model_l = LinearRegression()

        self.data = pd.read_csv("apartment_data.csv", encoding="UTF-8")

        self.x_for_l = self.data[["factor1_sqrt_m","factor2_stepfrommetro","factor3_nearby_to_center","factor4_rooms"]]
        self.y_total = self.data["predict"]

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.mae = None
        self.mse = None

    def learn_part(self):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_for_l, self.y_total, test_size=0.2, random_state=42
        )

        self.model_l.fit(self.x_train, self.y_train)


    def get_error(self):

        y_pred_test = self.model_l.predict(self.x_test)
        self.mae = mean_absolute_error(self.y_test, y_pred_test)
        self.mse = mean_squared_error(self.y_test,y_pred_test)


    def predict_new(self, features):
        return self.model_l.predict([features])


