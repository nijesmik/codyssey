import pandas as pd
from collections import Counter

# CSV 파일 읽기
df = pd.read_csv("prob-0101.csv")


# 기본 통계 정보 출력
def print_stats(df):
    print(f"총 영화 수: {len(df)}")
    print(f"배급사 수: {len(df['배급사'].unique())}")
    print(f"감독 수: {len(df['감독'].unique())}")

    # 출연진 수 계산 (중복 제거)
    actors = set()
    for cast in df["출연진"].dropna():
        actors.update([actor.strip() for actor in cast.split(",")])
    print(f"출연진 수: {len(actors)}")

    # 장르 수 계산 (중복 제거)
    genres = set()
    for genre in df["장르"].dropna():
        genres.update([g.strip() for g in genre.split(",")])
    print(f"장르 수: {len(genres)}")


# 영화 추천 클래스
class MovieRecommender:
    def __init__(self, df):
        self.df = df

    def recommend_by_actor(self, actor_name, limit=3):
        """특정 배우가 출연한 영화 추천"""
        movies = []
        for _, row in self.df.iterrows():
            if pd.notna(row["출연진"]) and actor_name in row["출연진"]:
                movies.append(
                    {"제목": row["제목"], "개봉일": row["개봉일"], "장르": row["장르"]}
                )
        return movies[:limit]

    def recommend_by_genre(self, target_genre, limit=3):
        """특정 장르의 영화 추천"""
        movies = []
        for _, row in self.df.iterrows():
            if pd.notna(row["장르"]) and target_genre in row["장르"]:
                movies.append(
                    {
                        "제목": row["제목"],
                        "개봉일": row["개봉일"],
                        "출연진": row["출연진"],
                    }
                )
        return movies[:limit]

    def recommend_by_director(self, director_name, limit=3):
        """특정 감독의 영화 추천"""
        movies = []
        for _, row in self.df.iterrows():
            if pd.notna(row["감독"]) and director_name in row["감독"]:
                movies.append(
                    {"제목": row["제목"], "개봉일": row["개봉일"], "장르": row["장르"]}
                )
        return movies[:limit]


# 메인 실행 코드
def main():
    # 기본 통계 출력
    print("=== 영화 데이터 기본 통계 ===")
    print_stats(df)
    print("\n")

    # 추천 시스템 초기화
    recommender = MovieRecommender(df)

    criterion = input("추천 기준을 선택하세요 (1: 배우, 2: 장르, 3: 감독): ")

    while isValidInput(criterion) == False:
        criterion = input(
            """잘못된 입력입니다. 추천 기준을 선택하세요 (1: 배우, 2: 장르, 3: 감독): """
        )

    movies = getMoviesByCriterion(recommender, criterion)
    if len(movies) == 0:
        print("검색 결과가 없습니다.")

    for movie in movies:
        print(movie)


def getMoviesByCriterion(recommender, criterion):
    if criterion == "1":
        actor_name = input("배우 이름을 입력하세요: ")
        movies = recommender.recommend_by_actor(actor_name)
    elif criterion == "2":
        target_genre = input("장르를 입력하세요: ")
        movies = recommender.recommend_by_genre(target_genre)
    elif criterion == "3":
        director_name = input("감독 이름을 입력하세요: ")
        movies = recommender.recommend_by_director(director_name)
    return movies


def isValidInput(criterion):
    c = int(criterion)
    if c < 1 or c > 3:
        return False
    return True


if __name__ == "__main__":
    main()
