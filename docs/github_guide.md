**Github Issue, PR(Pull Request) 사용법**
=====================================

- 이 문서는 Github의 Issue, PR(Pull Request)에 대한 소개와 사용법에 대해서 다룹니다.
- Github에서 Issue, PR은 하나의 묶음이라고 볼 수 있습니다.
- Issue는 작업할 내용을 기록하기 위해서 사용되고, PR은 작업한 내용을 main branch에 commit(merge)하기 위해서 사용됩니다.


### **0. Issue 생성하기**
- Github Issue는 "작업할 내용"을 기록하기 위해서 사용됩니다.
- "Github Repository -> Issues -> New Issues"를 통해서, 새로운 Issue 생성이 가능합니다.
<img src="">
- Issue에는 Background(작업을 소개하는 내용), Todo(작업할 내용), See also(참고할 내용), Assignees(작업하는 팀원 지정), Labels(이슈의 목적) 등을 이용해서 자세하게 기록할 수 있습니다.
<img src="">

### 1. F1 -> git clone -> Repository URL 입력
- 순차적으로 실행했을 때 결과는 아래와 같습니다.
<img src="">
<img src="">


### 2. F1 -> git fetch
- git fetch는 branch 목록들을 동기화 해주는 명령어 입니다.
- 이를 입력하면 main branch 뿐만 아니라, 다른 branch들도 원격으로 확인할 수 있습니다.
<img src="">

### 3. F1 -> git checkout -> branch 이름 입력
- git checkout은 branch를 변경하는 명령어 입니다.
- git checkout을 입력하면, 원격으로 접속 가능한 branch들이 조회되며 작업할 branch를 선택할 수 있습니다.
<img src="">
<img src="">


### 4. 이동한 branch에서 작업하기.
- 이동한 branch에서 작업을 진행하면 됩니다.
- 작업한 내용은 이동한 branch에서 commit하면서 기록합니다.


### **5. PR(Pull-Request) 생성하기**
- PR은 작업이 끝난 내용을 main branch에 merge(병합)하기 위해 필요합니다.
- "Github Repository -> Pull Request -> New pull request"를 통해서 새로운 PR 생성이 가능합니다.
<img src="">
- 생성한 PR의 예시는 아래와 같습니다.
- PR에는 Overview(작업한 내용), Issue Tags(작업한 Issue) 등을 이용해서 자세하게 기록할 수 있습니다.
- 특히 Issue Tags에서 "Closed | Fixed" 부분에 "#(Issue 번호)"를 입력하면, 해당 Issue가 자동으로 Close 됩니다.
<img src="">
---

### (Optional) VScode에서 모든 것을 제어하기
- 이전 과정을 통해서 Issue, PR에 익숙해졌다면 VScode Extension인 "GitHub Pull Requests"를 사용해서 모든 내용을 VScode 안에서 시도할 수 있습니다.
<img src="">