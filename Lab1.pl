
% Правила

least_popular([Game], Game).
least_popular([Game1, Game2 | Tail], LessPopular) :-
    players(Game1, Pl1),
    players(Game2, Pl2),
    Pl2 < Pl1,
    least_popular([Game2 | Tail], LessPopular).
least_popular([Game1, Game2 | Tail], LessPopular) :-
    players(Game1, Pl1),
    players(Game2, Pl2),
    Pl2 > Pl1,
    least_popular([Game1 | Tail], LessPopular).


% Присваивает X более популярную игру
more_popular(Game1, Game2, X) :-
    players(Game1, Players1),
    players(Game2, Players2),
    Players1 > Players2,
    X = Game1.
more_popular(Game1, Game2, X) :-
    players(Game1, Players1),
    players(Game2, Players2),
    Players1 < Players2,
    X = Game2.

% Находит самую популярную игру из списка
most_popular_game([Game], Game).
most_popular_game([Game1, Game2 | Tail], MostPopular) :-
    more_popular(Game1, Game2, MorePopular), % Более популярная игра
    most_popular_game([MorePopular | Tail], MostPopular). % Каждый раз оставляем более популярную

players_sum([], 0).
players_sum([Game | Tail], Current) :-
    players_sum(Tail, New),
    players(Game, Players),
    Current is New + Players.

% Получить список всех игр
all_games(Games) :-
    findall(Game, game(Game), Games).

% Получить список всех игр от разработчика
all_games_by_dev(Dev, GamesByDev) :-
    findall(Game, developer(Game, Dev), GamesByDev).

% Находит самую популярную игру ever
most_popular_game_all(Game) :-
    all_games(Games),
    most_popular_game(Games, Game).

% Суммарное количество игроков всех игр разработчика
players_by_dev(Dev, Players) :-
    all_games_by_dev(Dev, GamesByDev),
    players_sum(GamesByDev, Players).


% game(Название_игры).
game(zelda).
game(mario).
game(witcher3).
game(dark_souls).
game(overwatch).
game(cyberpunk2077).
game(minecraft).
game(fortnite).
game(league_of_legends).
game(dota2).
game(fallout4).
game(skyrim).
game(red_dead_redemption2).
game(gta5).
game(bioshock).
game(portal).
game(half_life).
game(starcraft).
game(heroes3).
game(diablo3).
game(among_us).

% developer(game, Разработчик_игры).
developer(zelda, nintendo).
developer(mario, nintendo).
developer(witcher3, cd_projekt_red).
developer(dark_souls, from_software).
developer(overwatch, blizzard).
developer(cyberpunk2077, cd_projekt_red).
developer(minecraft, mojang).
developer(fortnite, epic_games).
developer(league_of_legends, riot_games).
developer(dota2, valve).
developer(fallout4, bethesda).
developer(skyrim, bethesda).
developer(red_dead_redemption2, rockstar).
developer(gta5, rockstar).
developer(bioshock, irrational_games).
developer(portal, valve).
developer(half_life, valve).
developer(starcraft, blizzard).
developer(heroes3, nwc).
developer(diablo3, blizzard).
developer(among_us, innersloth).

% players(game, Количество_активных игоков).
players(zelda, 5000).
players(mario, 1000000).
players(witcher3, 75000).
players(dark_souls, 200000).
players(overwatch, 3000000).
players(cyberpunk2077, 500000).
players(minecraft, 15000000).
players(fortnite, 1500000).
players(league_of_legends, 8000000).
players(dota2, 6000000).
players(fallout4, 250000).
players(skyrim, 300000).
players(red_dead_redemption2, 400000).
players(gta5, 7000000).
players(bioshock, 10000).
players(portal, 1000).
players(half_life, 20000).
players(starcraft, 100000).
players(heroes3, 50000).
players(diablo3, 150000).
players(among_us, 5000000).


:- initialization(main).
main :-
    developer(Game, nintendo), players(Game, Players), \+ Players > 100000,
    write('Game from Nintendo <= 100k: '), write(Game), nl,

    players(Game1, Players1), Players1 > 100000,
    write('Game > 100k players: '), write(Game1), nl,

    most_popular_game_all(MostPopular),
    write('The most popular game: '), write(MostPopular), nl,

    all_games_by_dev(valve, GamesByDev),
    write('All games from Valve: '), write(GamesByDev), nl,

    most_popular_game(GamesByDev, MostPopularValve),
    write('The most popular game from Valve: '), write(MostPopularValve), nl,

    players_by_dev(valve, PlayersByDev),
    write('Total Valve games players: '), write(PlayersByDev), nl.
