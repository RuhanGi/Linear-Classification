NAME = lclass.exe
SRCDIR = src
SRCS =	main.cpp parse.cpp stats.cpp matrix.cpp predict.cpp \
		overload.cpp basis.cpp grad.cpp bayesian.cpp classifier.cpp
OBJS = $(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))

OBJDIR = obj
CXXFLAGS = -Wall -Wextra -Werror -g3 -MMD -MP

.SILENT:

all: $(OBJDIR) $(NAME)

$(OBJDIR):
	if not exist $(OBJDIR) mkdir $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	c++ $(CXXFLAGS) -c $< -o $@

$(NAME): $(OBJS)
	c++ $(CXXFLAGS) $(OBJS) -o $(NAME)

i: $(OBJDIR) $(NAME)
	-.\$(NAME) datasets/Iris/Iris.csv

t: $(OBJDIR) $(NAME)
	-.\$(NAME) datasets/titanic/train.csv

a: $(OBJDIR) $(NAME)
	-.\$(NAME) datasets/wine/winequality-red.csv

b: $(OBJDIR) $(NAME)
	-.\$(NAME) datasets/wine/winequality-white.csv

c: $(OBJDIR) $(NAME)
	-.\$(NAME) "datasets/automobile/imports-85.data"

f: $(OBJDIR) $(NAME)
	-.\$(NAME) "datasets/forest fires/forestfires.csv"

g: $(OBJDIR) $(NAME)
	-.\$(NAME) "datasets/genPoly-5.csv"

s: $(OBJDIR) $(NAME)
	-.\$(NAME) "datasets/synthCF.csv"

plot:
	python helper/plot.py res.csv

w:
	python helper/plotWeights.py Weights.csv

cf:
	python helper/plotClassif.py predictions.csv weights.csv

costs:
	python helper/costs.py cost_history.csv

clean:
	if exist $(OBJDIR) rmdir /s /q $(OBJDIR)

fclean: clean
	if exist $(NAME) del /q $(NAME)

re: fclean all

gpush: fclean
	git add .
	git commit -m "clean"
	git push