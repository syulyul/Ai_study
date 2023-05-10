# 1. 파이썬 기초
# 리스트
a = [38, 21, 53, 62, 19]
print(a)
print(a[0])

# 리스트 문자 출력
b = ['메이킷', '우진', '시은']
print(b)
print(b[0])
print(b[1])
print(b[2])

# 리스트 정수와 문자 출력
c = ['james', 26, 175.3, True]  # 여러가지 자료형을 한 번에 넣을 수 있음
print(c)

# 5번 문제 풀이
d = ['메이킷', '우진', '제임스', '시은']
# print(d[0], d[1])
# print(d[1], d[2], d[3])
# print(d[2], d[3])
# print(d)

print(d[0:2])
print(d[1:4])
print(d[2:4])
print(d[0:4])


# extend() 함수 사용하여 리스트 이어붙이기
a = ['우진', '시은']
b = ['메이킷', '소피아', '하워드']
a.extend(b) # 리스트 a에 리스트 b의 모든 요소를 추가하는 파이썬 리스트 메서드
print(a)
print(b)

c = ['우진', '시은']
d = ['메이킷', '소피아', '하워드']
d.extend(c)
print(c)
print(d)