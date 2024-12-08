**Github Issue, PR(Pull Request) 사용법**
=====================================

- 이 문서는 Github의 Issue, PR(Pull Request)에 대한 소개와 사용법에 대해서 다룹니다.
- Github에서 Issue, PR은 하나의 묶음이라고 볼 수 있습니다.
- Issue는 작업할 내용을 기록하기 위해서 사용되고, PR은 작업한 내용을 main branch에 commit(merge)하기 위해서 사용됩니다.
- Github branch를 분리함으로써 "팀원 간의 작업 환경 분리"를 하고, 이를 통해서 repository를 보다 체계적으로 관리할 수 있습니다.


### **0. Issue 생성하기**
- Github Issue는 "작업할 내용"을 기록하기 위해서 사용됩니다.
- "Github Repository -> Issues -> New Issues"를 통해서, 새로운 Issue 생성이 가능합니다.
![{23ED0AAD-86EC-4E6A-9630-4FFD07C65960}](https://github.com/user-attachments/assets/e1870cec-3792-46ec-bf02-ae1ff479d317)
- Issue에는 Background(작업을 소개하는 내용), Todo(작업할 내용), See also(참고할 내용), Assignees(작업하는 팀원 지정), Labels(이슈의 목적) 등을 이용해서 자세하게 기록할 수 있습니다.
![{C116D04F-B487-4238-9AC4-0FEA1BF0EAFC}](https://github.com/user-attachments/assets/7ffc822f-73c1-492f-9eed-f0dff6836159)


### 1. F1 -> git clone -> Repository URL 입력
- 순차적으로 실행했을 때 결과는 아래와 같습니다.
![{885C3C2F-1707-40C2-931D-DBEE8E1F00C5}](https://github.com/user-attachments/assets/8de425fc-88cf-4dd9-afb5-d8524b63f965)


### 2. F1 -> git fetch
- git fetch는 branch 목록들을 동기화 해주는 명령어 입니다.
- 이를 입력하면 main branch 뿐만 아니라, 다른 branch들도 원격으로 확인할 수 있습니다.
![{48960D99-4AEF-4D2A-8F47-F486015B2328}](https://github.com/user-attachments/assets/0774b752-508b-4a68-8a3a-aa8689cc26db)


### 3. F1 -> git checkout -> branch 이름 입력
- git checkout은 branch를 변경하는 명령어 입니다.
- git checkout을 입력하면, 원격으로 접속 가능한 branch들이 조회되며 작업할 branch를 선택할 수 있습니다.
![{32164ACB-8338-4DC4-BF5A-FAED3A85E40B}](https://github.com/user-attachments/assets/793cec9a-1da9-4406-b135-87b85a644133)


### 4. 이동한 branch에서 작업하기.
- 이동한 branch에서 작업을 진행하면 됩니다.
- 작업한 내용은 이동한 branch에서 commit하면서 기록합니다.


### **5. PR(Pull-Request) 생성하기**
- PR은 작업이 끝난 내용을 main branch에 merge(병합)하기 위해 필요합니다.
- "Github Repository -> Pull Request -> New pull request"를 통해서 새로운 PR 생성이 가능합니다.
![{A9E555EE-6933-4B31-A5F7-D84985B11E29}](https://github.com/user-attachments/assets/dc1ea4d8-5772-4c4e-9358-402ced2fa998)
- 생성한 PR의 예시는 아래와 같습니다.
- PR에는 Overview(작업한 내용), Issue Tags(작업한 Issue) 등을 이용해서 자세하게 기록할 수 있습니다.
- 특히 Issue Tags에서 "Closed | Fixed" 부분에 "#(Issue 번호)"를 입력하면, 해당 Issue가 자동으로 Close 됩니다.
![{5963EE2B-64F6-4368-9841-F7A7FAF3AE63}](https://github.com/user-attachments/assets/0dd05c93-096c-4911-8f45-e686f22691fe)
---

### (Optional) VScode에서 모든 것을 제어하기
- 이전 과정을 통해서 Issue, PR에 익숙해졌다면 VScode Extension인 "GitHub Pull Requests"를 사용해서 모든 내용을 VScode 안에서 시도할 수 있습니다.
![{34FEC3A5-FCB8-4CE4-A55F-72A00A8CB042}](https://github.com/user-attachments/assets/d108dfc8-5dc0-4e99-8ccd-4bf0960a3fbf)
- 또한 branch를 생성하면 "Commit graph"를 통해서 다른 branch의 기록도 확인할 수 있습니다.
![{5DEFD590-9875-49AB-8711-E4D798004593}](https://github.com/user-attachments/assets/16f1754b-1260-44bf-bcae-d31f848f811a)