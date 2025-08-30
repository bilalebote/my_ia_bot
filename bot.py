# bot.py
from transformers import AutoModelForCausalLM, AutoTokenizer

# موديل صغير مفتوح المصدر لتوفير مساحة
model_name = "nomic-ai/gpt4all-j"  

# تحميل الموديل والمفردات
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model_cache")

print("بوت الذكاء الاصطناعي جاهز! اكتب 'exit' للخروج.")

while True:
    user_input = input("أنت: ")
    if user_input.lower() == "exit":
        print("تم إنهاء البوت.")
        break

    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("بوت: " + response)
