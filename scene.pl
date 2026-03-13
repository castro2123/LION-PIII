% ===== Fatos Prolog gerados automaticamente =====

entity(a_laptop).
entity(a_page).
entity(a_table).
entity(a_woman).
entity(that).
entity(the_screen).

relation(that,talk,woman).
relation(woman,located,table).
relation(woman,talk,that).
relation(woman,with,laptop).

% ----- Regras genéricas -----
talking(X,Y) :- relation(X,talk,Y), relation(Y,talk,X).
near(X,Y) :- relation(X,near,Y); relation(Y,near,X).
located(X,Place) :- relation(X,_,Place).