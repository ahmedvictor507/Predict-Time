import tkinter as tk
import pandas as pd
from tkinter import messagebox
import joblib

model_saved_path = 'actual_time_model.pkl'
loaded_pipeline = joblib.load(model_saved_path)

def predict_clocked_time():
    item = entry_item.get()
    employee = entry_employee.get()
    operation = entry_operation.get()
    equipment = entry_equipment.get()
    qty = float(entry_qty.get())
    planned_hours = float(entry_planned_hours.get())

    new_data = {
        'Item': [item],
        'Employee': [employee],
        'Operation': [operation],
        'Equipment': [equipment],
        'Qty': [qty],
        'Planned Hours': [planned_hours],
    }

    new_data_df = pd.DataFrame(new_data)

    y_pred = loaded_pipeline.predict(new_data_df)

    messagebox.showinfo("Prediction Result", f"Predicted Clocked Hours: {y_pred[0]:.2f}")

app = tk.Tk()
app.title("Clocked Hours Prediction")

# Item input
tk.Label(app, text="Item:").grid(row=0, column=0, padx=10, pady=10)
entry_item = tk.Entry(app)
entry_item.grid(row=0, column=1, padx=10, pady=10)

# Employee input
tk.Label(app, text='Employee:').grid(row=1, column=0, padx=10, pady=10)
entry_employee = tk.Entry(app)
entry_employee.grid(row=1, column=1, padx=10, pady=10)

# Operation input
tk.Label(app, text='Operation:').grid(row=2, column=0, padx=10, pady=10)
entry_operation = tk.Entry(app)
entry_operation.grid(row=2, column=1, padx=10, pady=10)

# Equipment input
tk.Label(app, text='Equipment:').grid(row=3, column=0, padx=10, pady=10)
entry_equipment = tk.Entry(app)
entry_equipment.grid(row=3, column=1, padx=10, pady=10)

# Quantity input
tk.Label(app, text='Quantity:').grid(row=4, column=0, padx=10, pady=10)
entry_qty = tk.Entry(app)
entry_qty.grid(row=4, column=1, padx=10, pady=10)

# Planned hours input
tk.Label(app, text='Planned_Hours:').grid(row=5, column=0, padx=10, pady=10)
entry_planned_hours = tk.Entry(app)
entry_planned_hours.grid(row=5, column=1, padx=10, pady=10)

# Create and place the predict button
predict_button = tk.Button(app, text='Predict', command=predict_clocked_time)
predict_button.grid(row=6, column=0, padx=10, pady=10)

app.mainloop()