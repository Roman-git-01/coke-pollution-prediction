import tkinter as tk
from tkinter import messagebox
import threading
import sys
import os

# Проверка импорта
try:
    from detector_sim import main, post_data_incorrect
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Ошибка", f"Не удалось импортировать detector_sim.py\n{e}")
    sys.exit(1)

def run_main():
    try:
        main()
        # root.after(0, lambda: messagebox.showinfo("Готово", "main завершена"))
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Ошибка в main", str(e)))

def run_post_incorrect():
    try:
        post_data_incorrect()
        # root.after(0, lambda: messagebox.showinfo("Готово", "post_data_incorrect завершена"))
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Ошибка в post_data_incorrect", str(e)))

def on_start():
    threading.Thread(target=run_main, daemon=True).start()
    # messagebox.showinfo("Старт", "Функция main запущена в фоне")

def on_emergency():
    threading.Thread(target=run_post_incorrect, daemon=True).start()
    # messagebox.showinfo("Выброс", "Функция post_data_incorrect запущена в фоне")

root = tk.Tk()
root.title("Управление детектором")
root.configure(bg="white")
# Разворачиваем окно на весь экран
root.state('zoomed')
root.attributes('-fullscreen', True)
root.eval('tk::PlaceWindow . center')

btn_start = tk.Button(
    root,
    text="СТАРТ",
    font=("Arial", 40, "bold"),
    bg="#4CAF50",
    fg="white",
    activebackground="#45a049",
    activeforeground="white",
    bd=0,
    padx=40,
    pady=20,
    command=on_start
)
btn_start.place(relx=0.5, rely=0.4, anchor="center")

btn_emergency = tk.Button(
    root,
    text="ВЫБРОС",
    font=("Arial", 20, "bold"),
    bg="#f44336",
    fg="white",
    activebackground="#d32f2f",
    activeforeground="white",
    bd=0,
    padx=20,
    pady=10,
    command=on_emergency
)
btn_emergency.place(relx=0.5, rely=0.7, anchor="center")

root.mainloop()