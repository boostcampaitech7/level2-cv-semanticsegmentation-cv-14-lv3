1. 서버에 접속합니다. 앞으로 모든 Command는 Terminal에서 진행됩니다.

2. gdown을 이용해서 "diffusion 처리된 dataset"을 다운로드 받습니다.
	  - gdown은 Google Drive의 "링크 공유"를 이용해서 파일에 접근하고, 이를 빠르게 다운로드하는 library입니다. <br/>
	  a. ``` gdown 1tzhnQqsnx4InHsoRl96PNFFSWQyNpvFl ``` 을 입력합니다. <br/>
	  b. 만약 gdown을 실행할 수 없다는 error message를 얻는다면, ```pip install gdown```을 이용해서 gdown library를 다운로드 합니다. <br/>
	  c. 정상적으로 작동했다면, ".tar" 형식의 압축 파일이 서버에 설치됩니다.

3. 서버에서 압축 풀기 작업을 수행합니다. <br/>
	  - ``` tar -xvf data.tar.gz -C ./data ``` 를 입력합니다. <br/>
	  a. ```tar -xvf``` : ".tar" 파일의 압축 풀기를 수행합니다. <br/>
	  b. ```./data``` : 압축 풀기를 한 이미지들을 저장할 폴더 경로입니다. 사용자가 변경할 수 있습니다.
