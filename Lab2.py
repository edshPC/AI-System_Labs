from swiplserver import PrologMQI, create_posix_path
import re

PATH = create_posix_path("D:\ITMO\СИИ\Lab1\part_1.pl")

def all_games_from(prolog, dev):
    res = prolog.query(f"all_games_by_dev('{dev}', Res).")
    res = res[0]['Res']
    if len(res) == 0:
        print(f'В базе нет игр от {dev}')
    else:
        print(f'Игры от {dev}: {", ".join(res)}')

def most_popular_game_from(prolog, dev):
    res = prolog.query(f"all_games_by_dev('{dev}', GamesByDev), most_popular_game(GamesByDev, Res).")
    if not res:
        print(f'В базе нет игр от {dev}')
    else:
        res = res[0]['Res']
        print(f'Самая популярная игра от {dev} - {res}')

def more_popular_game(prolog, game1, game2):
    res = prolog.query(f"more_popular('{game1}', '{game2}', Res).")
    if not res:
        print(f'Не все запрашиваемые игры есть в базе')
    else:
        res = res[0]['Res']
        print(f'Более популярная игра - {res}')

def find_dev_of(prolog, game):
    res = prolog.query(f"developer('{game}', Res).")
    if not res:
        print(f'Запрашиваемой игры нет в базе')
    else:
        res = res[0]['Res']
        print(f'Разработчик {game} - {res}')

patterns = {
    r'мне нравятся игры от (.+)': all_games_from,
    r'какая самая популярная игра от (.+)': most_popular_game_from,
    r'какая игра популярнее: (.+) или (.+)': more_popular_game,
    r'кто разработчик (.+)': find_dev_of
}

def process_query(prolog, query):
    for pattern in patterns:
        match = re.match(pattern, query)
        if match is not None:
            patterns[pattern](prolog, *match.groups())
            return
    print("Неверный запрос")

with PrologMQI(prolog_path="C:\Program Files\swipl\\bin") as mqi:
    with mqi.create_thread() as prolog:
        prolog.query(f'consult("{PATH}")')
        while True:
            query = input('> ')
            if query == 'stop':
                break
            process_query(prolog, query.lower())
