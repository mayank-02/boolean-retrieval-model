from BooleanModel import BooleanModel

if __name__ == "__main__":
    model = BooleanModel("./corpus/*")

    print(model.query(input("Search >>> ")))