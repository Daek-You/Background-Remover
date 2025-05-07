# Background-Remover
![Remover](/images/MainImage.png)  

아이폰에서 사진 속 대상을 손가락으로 클릭하여, 피사체만 배경과 분리한 경험이 있을 겁니다. [누끼](https://namu.wiki/w/%EB%88%84%EB%81%BC)를 딴다고도 하죠. 이와 비슷한 효과를 낼 수 있게 만들었습니다. 사진과 대상 위치 정보를 받아, 해당 대상을 누끼딸 수 있게 해줍니다.  

## 기술 및 모델
### Segment Anything Model (SAM)

Meta AI에서 개발한 모델로, 이미지에서 객체를 분할(Segmentation)하기 위해 설계된 인공지능 모델입니다. 일반적인 세그멘테이션 모델과 다르게, SAM은 다음과 같은 특징이 있습니다.  
- <b>프롬프트 기반 세그멘테이션</b>: 사용자가 클릭한 지점을 입력으로 받아 해당 객체를 식별합니다.
- <b>범용성</b>: 다양한 종류의 객체를 인식할 수 있도록 대규모 데이터셋으로 학습되었습니다.
- <b>Zero-shot 학습</b>: 새로운 객체 유형을 별도 학습 없이 인식할 수 있습니다.  

<br>

SAM에는 `ViT-B`, `ViT-L`, `ViT-H` 이렇게 3가지 버전이 있습니다.  

| 특성 | ViT-B (Base) | ViT-L (Large) | ViT-H (Huge) |
|------|-------------|--------------|--------------|
| **모델 크기** | 91M 파라미터 | 308M 파라미터 | 636M 파라미터 |
| **파일 크기** | 358MB | 1.2GB | 2.4GB |
| **다운로드 URL** | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| **정확도** | 가장 낮음 | 중간 | 가장 높음 |
| **속도** | 가장 빠름 | 중간 | 가장 느림 |
| **메모리 요구사항** | 낮음 | 중간 | 높음 |
| **적합한 사용 환경** | 모바일 장치, 저사양 환경 | 일반 데스크톱, 중간 성능 요구 | 고성능 워크스테이션, 서버 |
| **IoU 점수(성능 지표)** | 86.8 | 88.8 | 90.4 |
| **추론 시간(상대적)** | 1x | 약 2.5x | 약 4x |  
