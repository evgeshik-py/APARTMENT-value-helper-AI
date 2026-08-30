from model import ApartmentValModel

model_apart = ApartmentValModel()
model_apart.learn_part()
model_apart.get_error()

print(f"MAE:{model_apart.mae}")
print(f"MSE:{model_apart.mse}")


new_pred = model_apart.predict_new([100,200,3,6])
print("Predict value on this apartment:""\n",new_pred)

