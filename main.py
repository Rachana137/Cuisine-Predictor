import argparse
import project2 as p2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="top N meals")
    parser.add_argument("--ingredient", action='append', type=str, required=True,
                        help="list of ingredients")

    args = parser.parse_args()
    dframe = p2.read_data('yummly.json')
    ingredients = p2.convert_string(dframe.ingredients, args.ingredient)
    matrix = p2.document_matrix(ingredients)
    cuisine, score = p2.predict_cuisine(dframe, matrix, args.ingredient)
    similar = p2.similarity(dframe, matrix, args.N)
    result=p2.output(cuisine, score, similar)
    print(result)

