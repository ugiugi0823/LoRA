

이 글은 설치 가이드가 아니라 결과를 개선하는 방법에 대한 가이드이고, 모델 학습에 어떤 옵션이 있는지 설명하며, 불량하거나 적은 수의 이미지만 사용하더라도 캐릭터를 훈련시킬 수 있는 방법이 담겨있음

이 글의 정보량이라면 챈럼들이 학습을 하며 생기던 의문점들에 대해 충분한 답을 해줄 수 있다고 생각해서 한글로 옮겨왔다.

읽기 전에 주의해야 할 점이, 이 글의 작성자는 한 명으로 추측되기 때문에 주관적인 견해가 들어가 있을 수 있다는 것이고(난 번역만 함), 특히 글에 써진대로 '적응형 옵티마이저의 팬'이기 때문에 저 부분에 대해서는 객관적으로 판단해 줬으면 함.

  

목차

- [서문](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%84%9C%EB%AC%B8)
- [KOHYA의 훈련 스크립트 설치](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#kohya%EC%9D%98-%ED%9B%88%EB%A0%A8-%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8-%EC%84%A4%EC%B9%98)
    - [Ubuntu 23.04 단계별 설치.](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#ubuntu-2304-%EB%8B%A8%EA%B3%84%EB%B3%84-%EC%84%A4%EC%B9%98)
        - [Python 3.10은 Ubuntu >=23.04에서 사용할 수 없습니다.](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#python-310%EC%9D%80-ubuntu---2304%EC%97%90%EC%84%9C-%EC%82%AC%EC%9A%A9%ED%95%A0-%EC%88%98-%EC%97%86%EC%8A%B5%EB%8B%88%EB%8B%A4)
            - [Python 빌드](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#python-%EB%B9%8C%EB%93%9C)
       [[로라 모든 것]]     - [특별한 venv 설정](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%8A%B9%EB%B3%84%ED%95%9C-venv-%EC%84%A4%EC%A0%95)
            - [Phantom 라이브러리 버전](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#phantom-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC-%EB%B2%84%EC%A0%84)
    - [bmaltais의 kohya_ss](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#bmaltais%EC%9D%98-kohya-ss)
        - [Ubuntu 23.04](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#ubuntu-2304)
- [신비한 닌자 스크롤](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8B%A0%EB%B9%84%ED%95%9C-%EB%8B%8C%EC%9E%90-%EC%8A%A4%ED%81%AC%EB%A1%A4)
    - [용어 해설](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9A%A9%EC%96%B4-%ED%95%B4%EC%84%A4)
    - [왜 이렇게 많은 스크립트가 있는 거죠? 혼란스러워요!](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%99%9C-%EC%9D%B4%EB%A0%87%EA%B2%8C-%EB%A7%8E%EC%9D%80-%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%EA%B0%80-%EC%9E%88%EB%8A%94-%EA%B1%B0%EC%A3%A0-%ED%98%BC%EB%9E%80%EC%8A%A4%EB%9F%AC%EC%9B%8C%EC%9A%94)
    - [왜 더 정확한 숫자가 없을까요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%99%9C-%EB%8D%94-%EC%A0%95%ED%99%95%ED%95%9C-%EC%88%AB%EC%9E%90%EA%B0%80-%EC%97%86%EC%9D%84%EA%B9%8C%EC%9A%94-)
    - [완벽한 LORA를 만들고 싶습니다. 모든 요소를 신중하게 배열하다가...](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%99%84%EB%B2%BD%ED%95%9C-lora%EB%A5%BC-%EB%A7%8C%EB%93%A4%EA%B3%A0-%EC%8B%B6%EC%8A%B5%EB%8B%88%EB%8B%A4)
        - [그렇다면 처음에 무엇을 해야 하나요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EA%B7%B8%EB%A0%87%EB%8B%A4%EB%A9%B4-%EC%B2%98%EC%9D%8C%EC%97%90-%EB%AC%B4%EC%97%87%EC%9D%84-%ED%95%B4%EC%95%BC-%ED%95%98%EB%82%98%EC%9A%94)
    - [드림부스는 이제 쓸모 없나요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%93%9C%EB%A6%BC%EB%B6%80%EC%8A%A4%EB%8A%94-%EC%9D%B4%EC%A0%9C-%EC%93%B8%EB%AA%A8-%EC%97%86%EB%82%98%EC%9A%94)
        - [그런데 드림부스보다 LoRA가 별로인 것 같아요.](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EA%B7%B8%EB%9F%B0%EB%8D%B0-%EB%93%9C%EB%A6%BC%EB%B6%80%EC%8A%A4%EB%B3%B4%EB%8B%A4-lora%EA%B0%80-%EB%B3%84%EB%A1%9C%EC%9D%B8-%EA%B2%83-%EA%B0%99%EC%95%84%EC%9A%94)
        - [하이퍼네트워크는 이제 무용지물인가요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%95%98%EC%9D%B4%ED%8D%BC%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EB%8A%94-%EC%9D%B4%EC%A0%9C-%EB%AC%B4%EC%9A%A9%EC%A7%80%EB%AC%BC%EC%9D%B8%EA%B0%80%EC%9A%94)
        - [임베드(텍스트 역전)는 이제 무용지물인가요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9E%84%EB%B2%A0%EB%93%9C(%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%97%AD%EC%A0%84)%EB%8A%94-%EC%9D%B4%EC%A0%9C-%EB%AC%B4%EC%9A%A9%EC%A7%80%EB%AC%BC%EC%9D%B8%EA%B0%80%EC%9A%94)
    - [두 모델 사이의 차이에서 LORA 만들기 ("증류")](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%91%90-%EB%AA%A8%EB%8D%B8-%EC%82%AC%EC%9D%B4%EC%9D%98-%EC%B0%A8%EC%9D%B4%EC%97%90%EC%84%9C-lora-%EB%A7%8C%EB%93%A4%EA%B8%B0--%22%EC%A6%9D%EB%A5%98%22-)
    - [LoRA를 체크포인트에 혼합하기](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#lora%EB%A5%BC-%EC%B2%B4%ED%81%AC%ED%8F%AC%EC%9D%B8%ED%8A%B8%EC%97%90-%ED%98%BC%ED%95%A9%ED%95%98%EA%B8%B0)
- [시작하기](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0)
    - [시작하려면 무엇이 필요한가요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8B%9C%EC%9E%91%ED%95%98%EB%A0%A4%EB%A9%B4-%EB%AC%B4%EC%97%87%EC%9D%B4-%ED%95%84%EC%9A%94%ED%95%9C%EA%B0%80%EC%9A%94?)
    - [캐릭터 난이도](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%BA%90%EB%A6%AD%ED%84%B0-%EB%82%9C%EC%9D%B4%EB%8F%84)
        - [여성 편향](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%97%AC%EC%84%B1-%ED%8E%B8%ED%96%A5)
        - [이전 훈련](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9D%B4%EC%A0%84-%ED%9B%88%EB%A0%A8)
        - [의도된 결과](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9D%98%EB%8F%84%EB%90%9C-%EA%B2%B0%EA%B3%BC)
        - [캐릭터 난이도와 설정](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%BA%90%EB%A6%AD%ED%84%B0-%EB%82%9C%EC%9D%B4%EB%8F%84%EC%99%80-%EC%84%A4%EC%A0%95)
        - [차량](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%B0%A8%EB%9F%89)
    - [시작 설정](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8B%9C%EC%9E%91-%EC%84%A4%EC%A0%95)
        - [최적의 설정](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%B5%9C%EC%A0%81%EC%9D%98-%EC%84%A4%EC%A0%95)
        - ["쉬운 설정" 
    - [어떤 모델에서 훈련해야 할까요?](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%96%B4%EB%96%A4-%EB%AA%A8%EB%8D%B8%EC%97%90%EC%84%9C-%ED%9B%88%EB%A0%A8%ED%95%B4%EC%95%BC-%ED%95%A0%EA%B9%8C%EC%9A%94?)
        - [NAI에 대한 법적 고려사항](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#nai%EC%97%90-%EB%8C%80%ED%95%9C-%EB%B2%95%EC%A0%81-%EA%B3%A0%EB%A0%A4%EC%82%AC%ED%95%AD)
        - [AnyLoRA](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#anylora)
        - [퍼리 커뮤니티의 질문](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%8D%BC%EB%A6%AC-%EC%BB%A4%EB%AE%A4%EB%8B%88%ED%8B%B0%EC%9D%98-%EC%A7%88%EB%AC%B8)
- [학습률 (LEARNING RATES)](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%95%99%EC%8A%B5%EB%A5%A0-learning-rates-)
    - [텍스트 인코더 학습률](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%9D%B8%EC%BD%94%EB%8D%94-%ED%95%99%EC%8A%B5%EB%A5%A0)
    - [Unet 학습률](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#unet-%ED%95%99%EC%8A%B5%EB%A5%A0)
    - [옵티마이저](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%80)
    - [옵티마이저](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%98%B5%ED%8B%B0%EB%A7%88%EC%9D%B4%EC%A0%801)
        - [Lion](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#lion)
        - [AdaFactor](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#adafactor)
        - [DAdaptation  
        - [Prodigy](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#prodigy)
        - [d](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#d)
        - [d_coef](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#d-coef)
        - [스케줄러](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8A%A4%EC%BC%80%EC%A4%84%EB%9F%AC)
        - [최소 SNR 감마](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%B5%9C%EC%86%8C-snr-%EA%B0%90%EB%A7%88)
    - [스케줄러](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8A%A4%EC%BC%80%EC%A4%84%EB%9F%AC1)
    - [네트워크 차원 (network dimensions)/랭크](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC-%EC%B0%A8%EC%9B%90--network-dimensions--%EB%9E%AD%ED%81%AC)
    - [네트워크 알파 (network alpha)](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC-%EC%95%8C%ED%8C%8C--network-alpha-)
    - [노이즈 오프셋](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%85%B8%EC%9D%B4%EC%A6%88-%EC%98%A4%ED%94%84%EC%85%8B)
    - [해상도](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%95%B4%EC%83%81%EB%8F%84)
    - [증강](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%A6%9D%EA%B0%95)
    - [Min. SNR gamma](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#min-snr-gamma)
    - [Scale Weight Norms](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#scale-weight-norms)
    - [시드](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%8B%9C%EB%93%9C)
    - [Loss](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#loss)
- [모델 테스트와 디버깅](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%AA%A8%EB%8D%B8-%ED%85%8C%EC%8A%A4%ED%8A%B8%EC%99%80-%EB%94%94%EB%B2%84%EA%B9%85)
    - [테스트의 중요성](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%85%8C%EC%8A%A4%ED%8A%B8%EC%9D%98-%EC%A4%91%EC%9A%94%EC%84%B1)
    - [강도 (생성 시간 기준)](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EA%B0%95%EB%8F%84--%EC%83%9D%EC%84%B1-%EC%8B%9C%EA%B0%84-%EA%B8%B0%EC%A4%80-)
    - [다양한 체크포인트에서 모델 테스트](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EB%8B%A4%EC%96%91%ED%95%9C-%EC%B2%B4%ED%81%AC%ED%8F%AC%EC%9D%B8%ED%8A%B8%EC%97%90%EC%84%9C-%EB%AA%A8%EB%8D%B8-%ED%85%8C%EC%8A%A4%ED%8A%B8])
    - [과적합](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EA%B3%BC%EC%A0%81%ED%95%A9)
        - [추가 네트워크 확장판을 사용하는 경우](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%B6%94%EA%B0%80-%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC-%ED%99%95%EC%9E%A5%ED%8C%90%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EA%B2%BD%EC%9A%B0)
    - [태그 디버깅](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%83%9C%EA%B7%B8-%EB%94%94%EB%B2%84%EA%B9%85)
- [희귀 캐릭터/OC 작성 가이드](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%ED%9D%AC%EA%B7%80-%EC%BA%90%EB%A6%AD%ED%84%B0-oc-%EC%9E%91%EC%84%B1-%EA%B0%80%EC%9D%B4%EB%93%9C)
    - [이미지 준비](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%A4%80%EB%B9%84)
    - [이미지 편집](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%8E%B8%EC%A7%91)
        - [Krita 간단한 스타터 팩](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#krita-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%8A%A4%ED%83%80%ED%84%B0-%ED%8C%A9)
    - [AI를 활용하여 도움 받기](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#ai%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%84%EC%9B%80-%EB%B0%9B%EA%B8%B0)
    - [AI 출력물을 입력으로 사용하기](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#ai-%EC%B6%9C%EB%A0%A5%EB%AC%BC%EC%9D%84-%EC%9E%85%EB%A0%A5%EC%9C%BC%EB%A1%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)
    - [GACHAMAX 임시 LoRA 방법 (마지막 수단!)](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#gachamax-%EC%9E%84%EC%8B%9C-lora-%EB%B0%A9%EB%B2%95--%EB%A7%88%EC%A7%80%EB%A7%89-%EC%88%98%EB%8B%A8--)
    - [일반적인 조언](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%9D%BC%EB%B0%98%EC%A0%81%EC%9D%B8-%EC%A1%B0%EC%96%B8)
    - [예술 기술을 가진 독자를 위한 팁](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#%EC%98%88%EC%88%A0-%EA%B8%B0%EC%88%A0%EC%9D%84-%EA%B0%80%EC%A7%84-%EB%8F%85%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-%ED%8C%81)
- [SAMPLE POWERSHELL SCRIPT (WINDOWS)](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#sample-powershell-script--windows-)
- [SAMPLE BASH SCRIPT (LINUX)](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1#sample-bash-script--linux-)

## 서문

이 안내서는 다른 LORA 훈련 안내서와 함께 공존하도록 되어 있습니다.  
이것은 "매우 건조한 과학 논문"과 "할아버지와 초보자를 위한 내 첫 번째 LoRA" 중간 지점을 목표로 하며, 옵션의 효과와 그들이 하는 일에 대한 설명을 최선을 다해 다루고, 또한 결과를 세부적으로 설명하고 내 팁과 노하우를 공유하고자 합니다.

가이드를 이미지로 가득 채우고 싶지만, 그러한 이미지의 양이 핸드폰이나 느린 인터넷 연결에서 가이드를 확인하는 데 문제가 될 수 있습니다. 그래서 최대한 설명만으로 작성하려고 노력하겠습니다.

LyCORIS 콘텐츠가 더 널리 퍼졌다고 생각하기 때문에 "일반적인 LoRA"를 LoRA-LierLa라고 부르기 시작하겠습니다. 평범한 "LoRA"를 남겨 두었다면 LoRA-LierLa에 대해 이야기한다고 가정하십시오. LyCORIS 유형은 LyCORIS-LoKR과 같이 LyCORIS-(유형)이라고 하며, 지정되지 않은 경우 대부분 또는 모든 유형에 적용되는 것으로 간주합니다.

저의 최종 목표는 누구나 캐릭터를 훈련시킬 수 있도록 하는 것입니다. 특히 낯선, 혹은 잊혀진 캐릭터들을 대상으로 합니다. 하지만 스타일이나 컨셉에 명백하게 유용한 내용은 언급하겠습니다.

CivitAI나 해당 서비스에 업로드할 때 결과를 개선할 수 있는 내용도 언급하겠습니다. 또한 자신의 특정한 취향에 맞게 모델을 조정하는 방법에 대한 조언도 제공하겠습니다.




## KOHYA의 훈련 스크립트 설치

[https://github.com/kohya-ss/sd-scripts](https://oo.pe/https://github.com/kohya-ss/sd-scripts "github.com/kohya-ss/sd-scripts (외부 사이트)") 를 복제하고 설치 지침을 따릅니다.  
!!! 위험 **WEBUI 설치 위치와 동일한 위치에 설치하지 마십시오! 새로운 파이썬 가상 환경을 만드십시오!**  
 스크립트는 다른 라이브러리 버전을 사용하며 웹 인터페이스를 망가뜨릴 수 있습니다. 설치 지침을 따른 후 이 글 하단의 bash(리눅스) 또는 powershell(윈도우) 스크립트를 가져와 필요한 대로 경로를 수정하여 Kohya의 스크립트를 실행합니다. 계속해서 테스트하는 과정에서 명령 줄 인수를 추가하겠습니다.

!!! 경고 일부 기능은 DAdaptation과 같이 별도로 추가되어야 합니다. 필요한 대로 표시될 것입니다.



### Ubuntu 23.04 단계별 설치.


#### Python 3.10은 Ubuntu >=23.04에서 사용할 수 없습니다.

sd-webui와는 크게 문제되지 않지만, Kohya 스크립트는 매우 특정한 버전을 약간 강제로 필요로 핮니다. 이로 인해 기본값 대비 약 5%의 속도 향상 효과가 나타납니다.  
별도의 Python 3.10 버전을 빌드하고 해당 버전을 venv로 사용해야 합니다. Python의 최신 tarball을 어딘가에 풀고 그 곳에서 터미널을 실행합니다.

- libssl-dev를 설치하십시오(아무 것도 다운로드하지 못할 수 있습니다)
- libbz2-dev를 설치하십시오(Kohya의 스크립트를 위해 필요한 기본 라이브러리입니다)
- 물론 build-essential과 같은 것들이 설치되어 있어야 합니다. tkinter를 위한 tk와 같은 라이브러리는 필요하지 않습니다(??).  
    sudo apt install libssl-dev libbz2-dev 명령으로 설치할 수 있습니다.

#####   

##### Python 빌드

./configure --enable-optimizations --with-ensurepip=install --prefix="/mnt/DATA/AI/Python310" --with-ssl-default-suites=openssl
make -j8 && make install   # 로컬 폴더이므로 root 권한이 필요하지 않습니다.



##### 특별한 venv 설정

Python 3.10.x가 모두 설정되었다면 해당 버전을 사용하여 가상 환경을 설정하려고 합니다.

/mnt/DATA/AI/Python310/bin/python3.10 -m venv venv   # 필요에 따라 경로를 변경하십시오.
. venv/bin/activate                                  # 이제 계속 진행할 수 있어야 합니다.



##### Phantom 라이브러리 버전

설치가 완료된 후 xformers가 남아 있을 가능성이 있습니다. 왜냐하면 Linux 시스템용으로 내장된 패키지가 PyTorch의 기본 설치와 호환되지 않기 때문입니다. 해결책은 기본 패키지를 설치한 다음 몇 가지 버전을 덮어쓰는 것입니다. 이들은 제대로 작동하는 것으로 확인되었습니다.

pip install torch==2.0.0 torchvision==0.15.1 -f [https://download.pytorch.org/whl/cu117](https://oo.pe/https://download.pytorch.org/whl/cu117 "download.pytorch.org/whl/cu117 (외부 사이트)")
pip install xformers==0.0.17



### bmaltais의 kohya_ss

[kohya_ss](https://oo.pe/https://github.com/bmaltais/kohya_ss#installation "github.com/bmaltais/kohya_ss (외부 사이트)")는 Kohya 스크립트와 자주 동기화되는 대안 설정으로 더 접근 가능한 사용자 인터페이스를 제공합니다.  
설치 지침을 보려면 해당 링크로 이동하십시오. 위와 마찬가지로 웹 인터페이스와 동일한 위치에 설치하지 **마십시오**.  
스크립트를 사용하는 것과 완전히 동일하며 Windows 사용자에게는 훨씬 편리하며 권장됩니다.  
옵션 이름과 내용은 대부분 동일할 것이므로 보다 번거롭지 않은 경험을 위해 사용하십시오.

#### Ubuntu 23.04

- Python 빌드 및 특별 venv 설정에 대해서는 동일한 단계를 사용합니다. 그러나 kohya-ss에서는 tkinter를 사용하므로 Python 3.10을 빌드할 때 tk8.6-dev가 설치되어 있어야 합니다(sudo apt install tk8.6-dev).  
    - make의 출력 중간 부근에 The necessary bits to build these optional modules were not found:라는 목록이 표시됩니다. tkinter가 없는지 확인하십시오.
- 설치 스크립트가 나머지 모든 것을 설정합니다.

##   

##   

## 신비한 닌자 스크롤

###   

### 용어 해설

|용어|설명|용어|설명|
|---|---|---|---|
|모델|"체크포인트"로도 알려져 있으며, 일반적으로 "가중치"를 포함한 단일 파일로 훈련 결과를 나타냅니다.|임베드|"텍스트 임베딩"(혹은 텍스트 반전)으로도 알려져 있으며, 텍스트 인코더만 훈련하는 예전 스타일입니다.|
|훈련|모델을 훈련하는 가장 흔한 용어|하이퍼네트워크|임베드와 유사하지만 Unet에 작용하는 것입니다.|
|신비한 닌자 스크롤|Kohya 스크립트의 전체 문서에 대한 재미있는 일본어 별명|주제|캐릭터, 물체, 차량, 배경 등을 모델로 훈련합니다.|
|Kohya|훈련 스크립트 및 기타 Stable Diffusion 관련 기술 개발자|스타일|특정한 미학을 재현하기 위해 모델을 훈련합니다.|
|webui|가장 일반적인 Stable Diffusion 생성 도구|컨셉|포즈나 구성과 같은 것을 재현하기 위해 모델을 훈련합니다.|
|익스텐션|플러그인과 유사한 WebUI의 확장 기능입니다. 웹 인터페이스의 확장 탭에서 추가할 수 있습니다.|훈련 세트|훈련 이미지와 태그를 결합한 것입니다.|
|Voldy|AUTOMATIC1111, webui의 저자|디스틸링|큰 모델에서 LORA를 추출하는 데 사용되는 가정용 용어입니다.|
|Unet|이미지 학습과 일부 알려진 결정/연관성 속성을 제어하는 시스템|과적합|"오버피팅"으로도 알려져 있으며, 모델이 훈련 세트를 지나치게 과도하게 재현하려고 할 때 발생합니다.|
|텍스트 인코더(TE)|프롬프트의 단어나 토큰을 AI가 이해할 수 있는 데이터로 변환하는 시스템|Deep-frying|생성된 이미지가 매우 채도가 높은 경우로, 일반적으로 높은 CFG 스케일의 결과입니다.|
|CLIP|일반적으로 우리가 훈련할 텍스트 인코더입니다. Stable Diffusion v2 모델은 OpenCLIP을 사용합니다.|Interrogator|이미지에서 발견한 항목의 태그를 제공하는 작은 AI입니다.|
|Net Dim / Rank|"랭크"로도 알려져 있으며, 모델의 전체 용량을 나타내며 보통 더 큰 파일로 표시됩니다.|추론|AI의 출력을 생성하는 과정입니다. "생성"의 동의어로 사용될 수 있습니다.|
|AI|사실, 이것은 AI라기보다는 "머신 러닝"에 가깝지만 비공식적으로 "AI"라고 부르는 것이 더 쉽습니다.|LyCORIS|LORA beYond Conventional methods, Other Rank adaptation Implementations, 다른 방식으로 LoRA를 구현하는 방법입니다.|
|드림부스|다른 유형의 훈련으로 큰 파일(2-4GB)이 생성됩니다.|LierLa|(또는 LoRA-LierLA) LoRA for _Li_n_e_a_r_ _La_yers, 원래 LoRA의 약어입니다.|
|LORA|이 안내서에서 다루는 훈련 유형입니다. 공식 철자는 "LoRA"이며 "Low Rank Adaptation"의 약어입니다.|C3Lier|(또는 LoRA-C3Lier) LoRA for _C_olutional _3_x3 Kernel and _Li_near Layers. 사용 빈도가 낮은 대체 LoRA 유형입니다.|
|바디 플랜|몸의 구성으로서 팔, 머리, 눈 등의 위치와 같은 부분이 모두 위치하는 곳을 말합니다.|||

###   

### 왜 이렇게 많은 스크립트가 있는 거죠? 혼란스러워요!

파워셀/배시 스크립트는 _Kohya_ 스크립트를 길고 귀찮은 인수들과 함께 실행하기 위한 것입니다. 이런 인수들을 수동으로 작성하는 것은 매우 번거롭습니다.  
무엇을 하고 있는지 알고 있다면 둘 다 필요하지 않습니다. 파워셸/배시 스크립트는 편의 기능이며, 때로는 혼동스러울 수 있습니다. 이 모든 내용을 수동으로 입력하고 변경하는 일을 상상해보세요!  
또한 이것은 더 숙련된 사용자가 훈련 일괄 작업을 자동화하는 기초가 됩니다.  
**오늘날에는 sd-webui와 비슷한 인터페이스를 가진 bmaltais의 UI와 같은 대안이 있으므로 스크립트를 무시할 수 있습니다.** 따라서 원하지 않는 한 스크립트에 신경쓰지 않아도 됩니다.

  

### 왜 딱 떨어지는 숫자가 없을까요?

모든 것이 대략적이고 추상적입니다. 예술의 퀄리티와 기대치 같은 주관적인 내용을 다루고 있기 때문에 명확한 측정값을 얻는 것이 어렵습니다. 결과가 매우 무작위로 나타나기 때문에 이러한 문제가 더욱 심각해집니다.

  

### 완벽한 LORA를 만들고 싶습니다. 모든 요소를 신중하게 배열하다가...

그렇게 하지 마세요. 멈추세요. 당신은 계획에 너무 많은 시간을 들이고 훈련에는 너무 적은 시간을 들이고 있습니다.  
AI가 정확히 무엇을 할지, 어떤 요소에 어려움을 겪을지, 주어진 이미지를 어떻게 받아들일지를 완전히 예측하는 것은 **불가능**합니다.

  

#### 그렇다면 처음에 무엇을 해야 하나요?

먼저 시험용 모델을 훈련하고 나중에 문제를 해결하세요. 이렇게만 해서 AI가 훈련 세트를 어떻게 이해했는지 알 수 있습니다. 몇 가지 기본 숫자를 사용하고 문제가 발생할 때 여기로 와서 해결하거나 개선하는 방법을 찾아보세요.  
훈련하는 데는 시간이 오래 걸리지 않으며, 운이 좋다면 처음 시도에서 성공할 수도 있습니다.


### 드림부스는 이제 쓸모 없나요?

아니요. LORA를 훈련하는 것이 경제적이고 일반적으로 더 빠르긴 하지만, 드림부스는 아직도 다음과 같은 용도로 유용합니다.

- 혼합에 사용할 전체 모델 생성.
- 훈련용 모델 생성(예: 시리즈 스타일을 위한 드림부스 생성 후 해당 드림부스에서 캐릭터 훈련).
- 일반적인 스타일 훈련.

####   

#### 그런데 드림부스보다 LoRA가 별로인 것 같아요.

사실은 그렇지 않습니다. 아마도 _좋은_ 드림부스를 기억하고 있는 것일 겁니다. 제대로 훈련되지 않은 드림부스는 LoRA와 마찬가지로 슬픈 결과를 낳습니다.  
LoRA 훈련은 훈련 모델에 엑세스하기 전에 떠돌았던 많은 나쁜 관행과 조언들을 쉽게(혹은 빠르게/일반적인 하드웨어를 사용하여) 알아차릴 수 있게 해주었습니다.  
잘 훈련된 LoRA는 유사한 범위(한 명/몇 명의 캐릭터, 스타일 등)에서 잘 훈련된 드림부스와 비교 가능하며, 현대적인 기술을 사용하면 분명히 더 나아지며, 훨씬 더 지속 가능합니다 (모든 캐릭터에게 각각 2/4GB를 사용하고 싶으세요?).

  

#### 하이퍼네트워크는 이제 무용지물인가요?

스타일에 대해서는 여전히 사용할 수 있습니다. 그러나 현재 가장 구식 기술 중 하나입니다. 특히 webui에서 유사한 (혹은 덜 조절 가능한) 효과를 위해 별도의 Unet을 선택할 수 있게 되어 더욱 구식 기술로 여겨집니다.  
하지만 좋아하는 하이퍼네트워크가 있다면 계속 사용해도 괜찮습니다. 여전히 유지되는 기술이기 때문입니다.

  

#### 임베드(텍스트 역전)는 이제 무용지물인가요?

일부 복잡한 프롬프트를 단순화하거나 부정적인 임베딩에 유용하다는 사실이 밝혀졌습니다.

  

### 두 모델 사이의 차이에서 LORA 만들기 ("증류")

신비한 닌자 스크롤에는 스크립트와 함께 제공되는 매우 유용한 도구에 대한 언급이 있습니다.  
이 개별 스크립트는 첫 번째 모델과 그 첫 번째 모델에서 훈련된 두 번째 모델 사이의 차이에서 LORA를 생성할 수 있게 해줍니다. 예를 들어 NAI와 HLL 또는 NAI와 Anything v3 사이의 차이에서 로라를 만들 수 있습니다.  
당연한 장점은 이후에 이 LORA를 다른 모델을 생성할 때 마치 다른 모델을 혼합하는 것처럼 조정 가능한 형태로 연결하여 유사한 효과를 얻을 수 있다는 것입니다. 개별 특수 효과나 상호간의 미묘한 차이만 가진 여러 모델은 LORA로 안전하게 변환하고 하드 디스크에서 쉴 수 있을 것입니다. 그냥 레몬에서 과즙만을 쭉 짜내듯이 말입니다.  
제 테스트 결과로는 원본의 약 95% 정도 성능을 보여주지만 훨신 더 빠른 교체가 가능합니다. 그리고 이미지 생성 실험을 할 때 훨씬 익숙한 방법입니다.

설치하기 위해 먼저 sd-scripts 디렉토리로 이동한 다음 터미널을 엽니다.  
!!! 경고 반드시 파이썬 가상 환경에 들어가야 합니다.  
그런 다음 다음과 같이 networks/extract_lora_from_models.py 스크립트를 실행합니다.

python networks/extract_lora_from_models.py --model_org <원래 모델 (예: NAI)> --model_tuned <주스를 추출할 모델> --save_to <출력 파일.safetensors> --dim <치수, 더 크면 더 크고 정확하게 됩니다, 제 경우에는 256이 잘 작동합니다>

그런 다음 생성된 LORA를 다른 모델처럼 로드할 수 있습니다.

[여기](https://oo.pe/https://pixeldrain.com/u/RjE3psRf "pixeldrain.com/u/RjE3psRf (외부 사이트)")에서 HLL2 모델의 샘플 추출 버전 LORA를 확인할 수 있습니다. 랭크는 192입니다. 제 경우에는 잘 작동합니다.

  

### LoRA를 체크포인트에 혼합하기

체크포인트라고 하면 AOM3 또는 Anything과 같은 큰 모델을 의미합니다.  
LoRA는 실제로 혼합 성분으로 사용할 수 있습니다. [supermerger](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1)와 같은 특정 확장 프로그램을 사용하면 다른 모델을 혼합하는 것처럼 LoRA를 체크포인트에 혼합할 수 있습니다.  
Supermerger는 조금 무서울 수 있지만 LoRA 탭에서 LoRA를 체크포인트 (보통 LoRA가 훈련된 **체크포인트**)에 혼합할 설정이나 더 빠르고 스크립트 없이 LoRA를 증류/추출할 설정을 찾을 수 있습니다 (또한 훈련된 체크포인트와 비교). 그런 다음 결과를 정상적으로 혼합할 수 있습니다.  
**원래 LoRA는 2/4GB의 용량을 가진 전체 모델로 제작되어야 할 예정이었습니다. 제 SSD가 차이값 (delta)만 사용하도록 만든 Kohya에게 감사드립니다.**  
이것은 요즘 나오는 믹스 모델에서 결과에 영향을 주기 위해 흔하게 사용되는 일반적인 기술입니다. 즉, 특정한 스타일이나 특징을 가진 체크포인트를 기본적인 훈련으로 조금 더 저렴하게(컴퓨팅 비용/전력 요금/하드웨어 관점에서) 더 빠르게 생성할 수 있으며, 여전히 좋은 결과를 제공합니다.

  

  

## 시작하기

  

### 시작하려면 무엇이 필요한가요?

- [Kohya 스크립트](https://oo.pe/https://github.com/kohya-ss/sd-scripts#windows-installation "github.com/kohya-ss/sd-scripts (외부 사이트)")를 설치해야 하며, 캐릭터의 경우 10에서 50개의 이미지, 스타일의 경우 100-4000개의 이미지, 컨셉의 경우 50-2000개의 이미지가 필요합니다.  
     _또는 Windows의 경우 bmaltais의 [kohya_ss](https://oo.pe/https://github.com/bmaltais/kohya_ss#installation "github.com/bmaltais/kohya_ss (외부 사이트)")를 사용하세요. 인터페이스가 있어서 스크립트가 필요하지 않고 노브만 조정하면 됩니다._이것은 꽤 좋은 UI이며, Kohya 스크립트와 충분히 빠르게 동기화되므로 믿을 수 있습니다.
    
- 적어도 6GB의 VRAM을 가진 비디오 카드가 필요합니다. 가능하면 nVidia 카드를 사용하여 CUDA를 지원하거나 [Google Colab](https://oo.pe/https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb#scrollTo=wmnsZwClN1XL "colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb (외부 사이트)")과 같은 온라인 컴퓨팅 서비스를 사용하세요. (이 Colab은 저의 작성이 아닙니다. 추천만 했습니다.).
    
- 훈련할 기본 모델이 필요합니다. 현재로서는 무엇보다도 애니메이션, 만화 등을 그릴 때 NAI 유출 모델과 실제 주제에 대한 Stable Diffusion 1.5를 주로 사용합니다. 다른 체크포인트도 괜찮지만 해당 체크포인트만을 사용하지 않으면 결과가 나빠지는 경향이 있습니다. [어떤 모델을 사용할까요?](https://oo.pe/https://rentry.org/59xed3#what-model-to-train-from "rentry.org/59xed3 (외부 사이트)")를 참조하여 자세한 내용을 확인하세요.
    
- 텍스트 편집기가 필요합니다. 메모장도 작동할 수 있지만 [Notepad++](https://oo.pe/https://notepad-plus-plus.org/ "notepad-plus-plus.org/ (외부 사이트)"), [VSCode](https://oo.pe/https://code.visualstudio.com/ "code.visualstudio.com/ (외부 사이트)"), Sublime Text, Vim, Emacs 또는 가지고 있는 편집기와 좀 더 프로그래밍 중심의 편집기를 권장합니다.  
    * bmaltais의 UI를 사용하지 않는 경우라도 어쨌든 괜찮은 텍스트 편집기가 있어야 합니다!
    
- 선택적으로 포토샵, [Krita](https://oo.pe/https://krita.org/ "krita.org/ (외부 사이트)") (권장), [GIMP](https://oo.pe/https://www.gimp.org/ "www.gimp.org/ (외부 사이트)"), [Paint.NET](https://oo.pe/https://getpaint.net/ "getpaint.net/ (외부 사이트)") 또는 가지고 있는 이미지 편집기가 필요합니다. 필요하지 않을 수 있지만 유용할 수 있습니다.  
    * 이미지를 편집해야 할 경우 약간 서툴러도 괜찮습니다. 너무 잘 그려야 할 필요는 없습니다.
    
- bmaltais의 사용자 인터페이스를 사용하지 않는 경우 [샘플 파워셸 스크립트(Windows)](https://oo.pe/https://rentry.org/59xed3#sample-powershell-script-windows "rentry.org/59xed3 (외부 사이트)") 또는 [샘플 배시 스크립트(Linux)](https://oo.pe/https://rentry.org/59xed3#sample-bash-script-linux "rentry.org/59xed3 (외부 사이트)")를 가져와서 스크립트를 실행할 때 기호에 맞게 편집하세요.
    
- 파인튜닝을 사용하는 경우 이미지에 캡션을 생성하는 인터로그레이터가 필요합니다. 웹 인터페이스와 함께 [이 확장 프로그램](https://oo.pe/https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor "github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor (외부 사이트)")을 사용하는 것을 권장합니다. 다양한 설정으로 이미지에 캡션을 일괄 생성할 수 있습니다.
    
- 인내심이 필요합니다. 첫 번째 시도에서 작동하지 않더라도 **짜증을 내지 않고** 해결할 수 있는 방법이 있을 가능성이 높습니다. 고품질은 시간이 걸립니다.
    

### 캐릭터 난이도

모든 캐릭터가 동일하게 학습되지는 않습니다 (말 그대로). 일부 캐릭터는 다른 캐릭터보다 더 정교하거나 화려할 수 있으며, 이러한 복잡성은 고려되어야 합니다.  
많은 훈련을 거친 후, 저는 캐릭터를 다음과 같은 범주로 분류할 수 있을 것 같습니다.

|범주|설명|Rank|훈련 유형|
|---|---|---|---|
|A|비교적 간단한 "실제적인" 여성 캐릭터로서 특이한 옷이나 디테일이 없는 경우입니다.|4-8|텍스트 임베딩, LoRA-LierLa, LyCORIS-LoKR|
|B|특이한 얼굴, 머리스타일, 색상 또는 의상을 가진 여성 캐릭터입니다.|4-16|LoRA-LierLa, LyCORIS-LoKR, LyCORIS-LoHa|
|C|매우 정교한 디테일이 있으며 날개, 갑옷, 꼬리와 같은 추가 요소가 있는 경우입니다.|8-32|LoRA-LierLa, LyCORIS-LoKR, LyCORIS-LoHa|
|D|인간형, 사람과 비슷한 특징을 가진 퍼리 캐릭터, 실제적인 인간 등이 포함됩니다.|32-64|LoRA-LierLa, LyCORIS-LoHa|
|E|더 특이한 유형, 측면에 팔과 다리가 두 개 있는 캐릭터 등이 포함됩니다.|64|LoRA-LierLa|
|F|동물, 더 동물적인 신체 구조를 가진 포켓몬, 드래곤, 인간형 로봇 등이 포함됩니다. 화려한 의상이나 갑옷을 입은 사람들도 이에 속합니다.|64-128|LoRA-LierLa|
|G|건담이나 EVA와 같은 인간형 메카, 외계인, 악마, 다리가 여러 개인 인간형 캐릭터 등이 포함됩니다.|128|LoRA-LierLa|
|H|비인간형 로봇, 몬스터, 곤충, 만화적이지 않은 드래곤, 실제적인 동물 등이 포함됩니다.|128-192|LoRA-LierLa|
|I|섬뜩한 괴물들|192-256?|LoRA-LierLa|

####   

#### 여성 편향

대부분의 기존 체크포인트는 여성 캐릭터에 대한 확고한 편향을 가지고 있어 남성 캐릭터는 훈련하기 약간 어려울 수 있습니다. 따라서 남성 캐릭터는 위에서 언급한 범주 중에서 D 이하 범주까지 ".5"로 간주할 수 있습니다.

####   

#### 이전 훈련

캐릭터에 대한 이전 지식이 체크포인트에 있는지 여부도 염두에 두어야 할 사항입니다. 이전 지식이 있는지 없는지에 따라 훈련 시간 및 난이도가 줄어들 수 있습니다.  
예를 들어, 두근두근 문예부의 Monika는 NAI에서 프롬프트만으로도 생성할 수 있으며 최근 모델에서도 그녀와 충분한 유사성을 유지합니다. 이는 Monika 모델을 훈련하는 것이 간단함의 경계선에 있으며 텍스트 반전이나 300스텝의 LoRA로 실현 가능하다는 것을 의미합니다. 이것은 그녀의 표준 의상을 생성할 때 가챠를 제거하는 데 관한 것입니다.  
로봇을 훈련하는 경우 건담은 Gespenst Type-RV보다 명백히 더 잘 알려져 있으므로 이미 활용할 만한 지식이 있습니다.

당신이 훈련할 체크포인트(훈련하거나 이미지를 생성하는 체크포인트)가 원하는 내용에 대한 일부 지식이 있는지 확인하고 동일한 태그를 사용하여 효과를 얻을 수 있는지 미리 확인하세요. 이전 지식을 의도적으로 무시하려면 다른 태그를 사용하세요.

####   

#### 의도된 결과

물론 **당신**이 모델을 사용하려는 목적도 훈련에 영향을 미칩니다.  
일부 캐릭터 모델은 캐릭터를 벗은 상태로 생성하는 것에 중점을 둔 경우가 많아 정확성이 필요하지 않을 수 있습니다. CivitAI 등에 모델을 업로드할 때 이 점을 염두에 두세요. 사람들은 당신이 의도하지 않은 부분의 정확성을 기대할 수 있습니다.(모든 사람이 성적 콘텐츠를 다운로드하거나 훈련하는 것은 아닙니다). 음란물은 피하세요.  
그러나 그 외에도, 그것이 **의도된 결과**라면, 당신이 실제로 중요하게 생각하는 것은 얼굴을 올바르게 만드는 것이므로 매우 공격적으로 훈련할 필요가 없습니다.  
옷, 머리스타일, 얼굴 세부사항, 체형 등에 더 높은 정확성이 필요한 경우 더 긴 시간과 최적화된 설정 및 덜 게으른 훈련 세트로 훈련해야 합니다. 다양한 자세와 세부 사항의 정확성을 보장하기 위해요.

포켓몬이나 퍼리 캐릭터와 같은 것을 훈련하려면 모델을 어떻게 사용할지 생각해야 합니다. 생물을 인간형으로 만들고 싶다면 약간 더 전략적으로 생각해야 합니다. 학습률을 낮추고 에포크를 줄이고 가중 블록을 줄이면 됩니다. 그러나 정확성을 유지하려면 더 많은 에포크와 더 공격적인 학습률 등이 필요합니다. 그러므로 체형과 세부 사항이 제대로 학습되고 적용됩니다.

####   

#### 캐릭터 난이도와 설정

더 복잡한 캐릭터 (범주 >=E)는 일반적으로 **더 높은 차원/랭크** 또는 **더 높은 학습률** 및 **더 많은 훈련 시간**을 활용하는 것이 좋습니다. 낮은 랭크는 복잡한 체형 계획을 완전히 이해하지 못하는 것으로 보입니다. 따라서 더 공격적인 훈련이나 적합한 시간이 필요하며 결과가 그렇게 좋지 않을 수 있습니다.  
**특정 유형의 LyCORIS**를 사용할 때 이 점을 고려하세요. LoKR과 같은 옵션은 D 이상의 범주에 속하는 캐릭터에게는 좋지 않습니다. 축소된 랭크와 크기로 인해 필요한 "표현 수준"을 유지할 수 없기 때문입니다. 이러한 캐릭터에게는 더 많은 시간이 필요합니다.  
128 랭크의 표준 LoRA-LierLa 모델은 G 범주까지의 캐릭터를 홀드할 수 있습니다.  
**!!! 경고** 하지만 반대 또한 말이 됩니다. A 또는 B 범주의 캐릭터에 대해 64 랭크를 사용하면 디스크 공간과 컴퓨팅 리소스가 낭비됩니다. 낮은 범주의 캐릭터에 대해서는 낮은 랭크를 사용하세요.

####   

#### 차량

이를 철저히 테스트하지는 않았지만, 차량은 보통 C-G 범주 사이에 위치할 것으로 보입니다. 이는 차량의 복잡성과 세부 사항 수준에 기반합니다.  
차량을 훈련하는 경우 가능한 실제감 있게 만들고 싶을 것입니다. 따라서 큰 랭크 쪽으로 조정하는 것이 좋을 수 있습니다. 그러나 충분한 훈련 시간이 주어진 경우 64 랭크로 LyCORIS-LoKR을 사용하여 실제감 있는 차량을 얻는 것도 괜찮을 것으로 보입니다. 천천히 조리해보세요.  
물론 의도된 결과 정책을 염두에 두세요. 자동차를 애니메이션으로 변환하려면 약간 줄여야 할 수 있지만, 완전한 실제감을 위해 128 랭크 Lora-LierLa로 시도할 가치가 있을 수 있습니다. 그러나 지나치게 디테일하고 실제감 있게 만들 필요가 없다면, 이 경우 32와 64로 줄이는 것을 권장합니다. 어떻게 나오는지 확인하세요.

###   

### 시작 설정

**!!! 정보** 이 문서는 작업 중이며, 범위 및 매개변수는 더 많은 모델을 훈련함에 따라 변경될 수 있습니다. 이것은 일반적인 훈련을 위한 표준 범위입니다. 각 훈련 세트마다 다른 요구 사항이 있지만, 이 범위는 안전하게 보입니다.

|범주|이미지 수|Net Dim/Rank|알파|Unet 학습률|TE 학습률|정규화|총 스텝|해상도|
|---|---|---|---|---|---|---|---|---|
|캐릭터 (좋은 입력)|35-100|32-96|1-랭크|0.0001|0.00005|없음|1000+|512-768|
|캐릭터 (나쁜 입력)|15-30|32-64|1-랭크|0.0001|0.000045|예|1600+|512-768|
|스타일|100-10000+|96-192|1에서 (랭크/2)까지 랭크|0.0001|0.00004|없음|~3000+|576-768|
|컨셉|50-2000|8(!)-128|1에서 (랭크/2)에서 랭크|0.0001|0.000045|미정|미정|512-768|
|내 현재 설정 (캐릭터) |50-100|64|64|적응형|적응형|둘 중 하나|~1000|512-576|

총 단계는 옵션, 이미지 수, 품질 등에 따라 달라지며, 일반적으로 무언가가 유지되기 위해서는 1000스텝 이상이 필요합니다.

- 더 높은 랭크는 "비정상적인" 캐릭터와 이미지 수가 엄청난 경우(10000+)에 도움이 될 수 있지만, 그 외의 경우에는 어느 정도 이상의 성과를 제공할 것입니다.
- 알파는 랭크보다 높지 않아야 합니다.
- 스텝 수에 대한 자세한 내용은 [스텝/에포크 수](https://oo.pe/https://rentry.org/59xed3#number-of-stepsepochs "rentry.org/59xed3 (외부 사이트)")를 참조하세요.

이전에는 에포크보다 반복 횟수가 더 많아야 한다는 조언이 있었지만, 이제는 그렇지 않다고 보입니다. 따라서 여러 주제나 개념을 균형있게 조정하여 숫자를 일치시키는 경우를 제외하고는 에포크를 사용하세요. 그렇지 않으면 1회 반복하고 약 1000 단계에 도달할 수 있는 충분한 에포크를 실행하세요.

**스크립트를 한 번 실행하고 수행할 총 단계를 기록**해두는 것이 좋습니다. 그러면 정확한 숫자로 작업할 수 있습니다.  
!!! 주의 **적어도 일부 에포크로 나눠 저장하는 것이 좋습니다**. 그렇게 하면 진행 상황을 추적하거나 디버깅하는 데 사용할 수 있는 스냅샷을 만들 수 있습니다. 저는 4 에포크마다 스냅샷을 저장합니다.

####   

#### 최적의 설정

좋은 결과를 얻기 위해 훈련 스크립트에 집어넣을 수 있는 최적의 숫자는 없습니다.  
기본 설정으로 좋은 결과를 얻을 수도 있고, 기본 설정으로 재앙적인 결과를 얻을 수도 있습니다. 실제로 시도해보기 전까지는 정확히 알 수 없습니다.  
_완벽한_ 모델을 얻으려면 여러 번의 시도, 문제 해결 및 인내가 필요합니다. 이 모든 것은 주제의 복잡성, AI에게 얼마나 명확한지, 제대로 태그되었는지 등에 기반합니다. 물론 개인의 기준에도 의존합니다.  
현재는 적응형 옵티마이저를 사용하여 일관된 설정 세트를 유지하기가 훨씬 쉬워졌지만, 여전히 처음 시도가 크게 성공하지 못하는 경우 몇 가지 설정을 조정해야 합니다. 주로 에포크, 랭크/알파, 그리고 해상도일 것이지만, 이러한 경우 대부분의 문제는 훈련 세트의 문제입니다. 따라서 많은 추측 작업이 줄어듭니다.

다음 섹션의 설정은 저에게 잘 작동했습니다.

####   

#### "쉬운 설정"🏵️🏵️

많은 시도와 실험 끝에 DAdaptAdam을 사용하는 것이 가장 간단하고 쉬운 방법임을 확인했습니다.  
태그를 사용하거나 사용하지 않고도 좋은 결과를 얻을 수 있으며, 평균적으로 1000개의 총 단계와 50-100개의 이미지를 사용합니다. 이 방법은 어떤 캐릭터에 대해서도 매번 작동하며, 많은 캐릭터를 훈련시켜봤을 때도 효과적입니다.  
훈련 세트가 조잡하거나 AI가 고집스러워서 배우지 않는 경우에는 작동하지 않는 경우가 있었지만, 제 경우에는 일관된 결과를 얻었습니다.

|Setting|Value|
|---|---|
|Optimizer|DAdaptAdam|
|Learn rate (Unet)|1.0|
|Learn rate (TE)|1.0|
|Total steps|Around 1000, adjust epochs to fit your source images|
|scheduler|Constant|
|Alpha|1|
|Net dim (Rank)|64 (32 will be fine for characters and concepts too)|
|Resolution|512 (576 is also fine)|
|Bucket size|min:320, max:768-1024|
|min_snr_gamma|5|
|Max token length|225|
|Optional|--flip-aug|
|optimizer_args|--optimizer_args "decouple=True" "weight_decay=0.01" "betas=0.9,0.999"|

이 같은 설정을 Prodigy로 사용한 경우(옵티마이저 인수만 변경)에도 매우 좋은 결과를 얻었습니다. Prodigy는 미래지향적인 것 같아서 향후 변경할 계획입니다.

이 옵션 조합을 발견한 이후로 훈련한 모든 결과물이 상당히 좋았습니다. 드물게 2 에폭(200 단계) 더 필요한 경우도 있었지만, 그 외에는 거의 훌륭한 결과를 얻었습니다. 이 설정을 사용하면 입력 이미지가 끔찍하지 않다면 모델이 괜찮게 생성될 것입니다.

또한 이 설정으로 우연히 **스타일**을 훈련했는데, 예상 밖으로도 _매우 잘 작동했습니다_. 그래서 이 설정은 캐릭터 뿐만 아니라 다양한 스타일에도 적용할 수 있습니다.

이 설정은 드림부스 스타일(키워드만 사용) 및 파인튠 스타일(태그 사용)에도 잘 작동하므로 최소한의 노력으로 LoRA를 생성하려면 이것을 사용하시면 됩니다.

###   

### 어떤 모델에서 훈련해야 할까요?

**!!! 정보** "training from"라는 용어는 모델에서 훈련을 재개하는 것을 의미하며, "체크포인트"로도 알려져 있습니다.  
**간단한 답변**: 2D(애니메이션, 만화, 스케치 등)의 경우 NAI를, 현실/기타의 경우 SD1.5를 통해 훈련하면 됩니다.  
일반적으로 **"공유된 계보(shared ancestry)"가 많은 모델에서 훈련하는 것이 좋습니다**. 예를 들어 대부분의 알려진 혼합 모델은 NAI를 포함하거나 NAI에서 파생되었으므로, NAI에서 모델을 훈련하면 모든 혼합 모델과 호환될 수 있습니다.  
하지만 너무 멀리 나가면 "혼합성 치매(mixing dementia)"에 영향을 받을 수 있습니다. 따라서 SD1.5에서 2D를 훈련하면 혼합 모델에서 결과물이 좋지 않을 수 있습니다.

- 혼합 모델에서 훈련할 수는 있지만, 다른 모델로 생성할 때 효과를 예측하기 어렵습니다.
- 태깅은 호환되어야 합니다 (NAI 태그를 e6나 Waifu Diffusion 태그로 훈련된 모델에 사용하지 마세요. "잘못된 위치를 가리키게 되어" 미지의 문제를 일으킬 수 있습니다).  
    - 우연한 일부 경우가 있을 수 있지만, 실험적인 목적이 아니라면 피하는 것이 좋습니다.  
    AnyLoRA나 같은 최신 옵션을 사용할 수도 있습니다. 여기서는 개인적인 기호가 중요한 역할을 하지만, 여전히 골드 스탠다드로서 NAI를 추천합니다. 실제로 기술적으로 기능은 하더라도 AnyLoRA를 사용하면 내 모델이 NAI로는 그렇게 좋게 나오지 않는 경우가 있습니다.

####   

#### NAI에 대한 법적 고려사항

귀하의 우려가 도덕적인 측면에서 더 크다면, SD1.5나 SDXL을 사용하여 훈련하세요(완전히 사용 가능한 상태가 된 이후에). 결과물은 그렇게 좋게 나오지 않을 수 있지만 어느 정도 작동할 것입니다. 솔직히 말해서, 귀하에게 이것이 많은 걱정을 준다면 이미 피해가 발생한 상황이므로 NovelAI에 구독하거나 익명 기부를 보내는 것이 좋습니다. 그렇게 하면 모두에게 윈-윈이 됩니다.  
귀하의 우려가 상업화와 저작권 측면에서 더 크다면, 변호사와 상담해야 할 것입니다. 그러나 전체적으로 이 문제는 어두운 면이 있다고 생각합니다. 다만 개인적으로는 이 모든 것이 재미있는 일일 뿐이에요.

####


#### AnyLoRA

AnyLoRA는 훈련에 사용할 수 있는 새로운 체크포인트입니다.  
솔직히, 몇 번 시도해보았지만 결과물이 보통 좋게 나오지 않았습니다. 그러나 그럼에도 불구하고 결과물은 여전히 인식 가능하며 "부서지지 않은" 상태였습니다. **일부 경우에는 괜찮게 나오는 경우도 있습니다**. 따라서 두 가지 방식을 모두 시도하고 여러 체크포인트에서 어떤 것이 가장 좋아 보이는지 선택하는 것이 좋습니다 ([여러 체크포인트에서 모델을 시험해보고 계신 거죠?](https://oo.pe/https://rentry.org/59xed3#test-your-model-with-multiple-checkpoints "rentry.org/59xed3 (외부 사이트)")).  
개인적으로는 animefinal-full (NAI)을 2D(애니메이션 또는 만화)의 골드 스탠다드로 여기며, 적어도 지금은 그렇습니다 (SDXL이 이러한 상황을 변경할 수도 있습니다).  
또한 AnyLoRA가 많은 수의 믹스 모델에 포함되어 있지는 않은 것으로 보입니다. 이로 인해 본질적인 가치에 대해 의문을 품게 되었습니다. 언어 비유로 표현하자면, 지역 방언으로 에세이를 쓰려는 것과 비슷한 상황일 것입니다.

####   

#### 퍼리 커뮤니티의 질문

퍼리 커뮤니티로부터 계속해서 이 질문을 받게 되었고, 실험을 진행해보아야 했습니다만, 대부분의 경우 NAI (animefinal-full)는 퍼리 캐릭터를 배우는 데 충분합니다.

- 네, 이 캐릭터에게 애니메이션 눈을 줄 수는 없습니다. **(너무 적은 단계를 거친 경우나 이미 캐릭터가 애니메이션 눈을 가지고 있는 경우에는 예외)**.
- **해상도를 약간 높이는 것**은 비늘, 가시 등과 같은 미세한 디테일에 도움이 될 수 있습니다.
- **포켓몬** 및 비슷한 만화 동물은 NAI만 사용해도 완벽하게 작동합니다. 이미 많은 예시가 있습니다.
- NAI를 기본으로 사용함으로써 모델을 다른 개념과 **더 호환 가능하게** 만들 수 있습니다.

![[Pasted image 20240214110620.png]]

  

이 LoRA는 Prodigy로 훈련되었으며, 1200 스텝, 90 개의 이미지, 512 해상도, booru 태그가 지정되었으며, NAI에서 기반을 두었습니다 (위에서 제시한 내 설정을 참조하고, Prodigy 섹션에서 기본 Prodigy args를 복사하십시오). 이 이미지는 inpainted되지 않았으며 (adetailer가 실행되지 않음), 크기를 절반으로 줄인 것을 제외하고는 어떠한 방식으로도 편집되지 않았습니다. 볼 수 있듯이 "어떤 정도의 인간 형상"을 가지고 있으므로 문제없이 작동할 것입니다. 다양한 포즈를 얻을 수 있으며 옷도 적절한 비율로 생성할 수 있습니다. 이 예시는 배경이 없는 경우입니다(배경을 지정하지 않았음). 그럼에도 불구하고 이 작업을 수행할 수 있습니다. 이를 AOM3에 넣어 해당 스타일의 특징적인 스타일로 캐릭터를 생성할 수 있습니다.

이제, 아마도 다음과 같은 것을 원하실 것입니다:

- 더 현실적인 특징과 더 나은 포즈를 위해 fluffyrock과 같은 모델을 사용하려는 경우 (뒤로 굽은 무릎이나 다수의 팔 등)
- 물리적으로 정확한 나사와 볼트 및 이러한 부품들을 티타일레이션(적잖은 문제...) 목적으로 조합하는 경우
- 저자 스타일 태그를 사용하려는 경우? (퍼리와 AI 사이에 이런 것이 있지만, 저는 이런 매력을 제대로 이해하지 못했습니다)
- fluffyrock만 사용하여 모든 인기있는 혼합, 개념 LoRA 및 기타 등을 무시하고 이미지를 생성하고 싶은 경우
- 엄격하게 e612 태그와 해당 효과만 사용하려는 경우

개인적으로는 더 특이한 베이스 모델을 사용하면 생태계가 분열되는 경향이 있으므로 당신이 원하는 대로 되기 전까지는 권장하지 않겠습니다.  
**나는 퍼리가 "서로의 어딘가를 부딪히는" 이미지를 생성해본 적이 없기 때문에 (내가 글을 써야 하는 부분...) 여기서의 결과는 다를 수 있습니다.**

##


## 학습률 (LEARNING RATES)

[Unet/TE 강도 그리드 0.1, 0.25, 0.5, 1 및 2](https://oo.pe/https://i.imgur.com/OFanVlM.jpeg "i.imgur.com/OFanVlM.jpeg (외부 사이트)"), [Unet/TE 강도 그리드 1.0, 1.2, 1.4, 1.5, 1.6, 1.8](https://oo.pe/https://i.imgur.com/dktVoXT.jpg "i.imgur.com/dktVoXT.jpg (외부 사이트)")  
-> 학습률의 영향을 보여주는 그림. 이 모델은 Unet 훈련이 충분하지만 약간 더 많은 TE 훈련이 필요한 것으로 보입니다. Unet 1.0 - TE 1.5는 정확하지만 치비(chibi) 스타일은 아닙니다. 이는 다음 훈련이 1.5의 LE 비율로 진행되면 더 나은 결과를 가져올 것입니다.<-

**!!! 정보** 지수 표기법 숫자를 받아들일 수 없다면, Kohya의 스크립트는 실수도 허용합니다. 따라서 "1e-4"는 "0.0001"로, "5e-5"는 "0.00005"로 표현됩니다. 취향에 따라 결정하십시오.

**!!! 정보** DAdaptation (DAdaptAdam/DAdaptLion) 또는 Prodigy와 같은 적응형 옵티마이저를 사용하는 것이 권장됩니다. 적응형 옵티마이저를 사용하면 학습률을 조정하는 추측 작업이 사라지며, Unet을 손상시키는 위험 없이 에포크를 더 많이 훈련할 수 있습니다.

적응형 옵티마이저를 사용할 수 없는 경우, 학습률을 제어하는 두 가지 방법이 있습니다.

|옵션|값|효과|
|---|---|---|
|--learning_rate|0.005-0.0001|학습률의 마스터 조절. 다른 두 값을 동일한 값으로 설정합니다.|
|--unet_lr|0.0001-0.005|Unet의 학습률을 설정합니다. 모델의 가장 민감한 부분으로, 너무 높게 설정하지 마십시오.|
|--text_encoder_lr|0.00001-0.00005|텍스트 인코더의 학습률을 설정합니다. 모델의 언어 처리 부분입니다. Unet의 값보다 훨씬 낮게 설정하세요.|

이게 무슨 뜻인가요?  
만약 신경 쓰지 않는다면, 다른 두 값을 설정하려면 --learning_rate를 설정하십시오.  
그렇지 않으면 각각 설정하십시오. 다른 두 값을 설정한 경우에는 --learning_rate를 지정하는 것이 중복되므로 제가 Unet LR과 동일한 값을 설정했습니다.

각 훈련 구성 요소가 어떤 역할을 하는지 알아보려면 아래 내용을 읽어보세요.

###   

### 텍스트 인코더 학습률

텍스트 인코더는 생성 중에 텍스트 프롬프트를 어떻게 해석하고 훈련할 때 "뉴런"에 항목을 연결하는 역할을 합니다.

Kohya 스크립트의 설명서는 텍스트 인코더에 5e-5를 사용하는 것을 제안합니다. 만약 이 옵션이 지정되지 않으면, --learning_rate의 값을 사용합니다.  
동일한 시드에서 정확히 같은 훈련 세트를 사용하여 이 옵션만 변경한 모델을 테스트해보면, 세부 사항이 더 잘 분리되는 것처럼 보입니다.

- LE 학습률을 낮추면 물체를 더 잘 분리하는 데 도움이 될 수 있습니다. 생성물에 원하지 않는 물체가 나타나면 이를 낮추는 것이 좋습니다.
- 프롬프트에 가중치를 많이 주지 않고 물체를 나타내기 어렵다면, **너무 낮게 설정한 것**일 수 있습니다.

텍스트 인코더는 태그나 토큰을 Unet의 데이터와 연결하는 역할을 합니다. 물론, 물체의 더 많은 정의는 프롬프트에 대한 더 많은 제어를 의미합니다.  
Unet 자체는 태그의 지도 없이도 어떻게 요소를 조합할 수 있는지를 학습할 수 있지만, TE를 잘 훈련하면 더 세밀한 기능을 가질 수 있습니다. 다시 말해, 더 많은 프롬프트 제어가 가능하게 됩니다.

###   

### Unet 학습률

Unet은 시각적 기억의 대략적인 동등물로 작용합니다. 또한 학습한 요소들이 어떻게 상호 관련되며 구조 내에서의 위치를 가지고 있는지에 대한 일부 정보를 가지고 있는 것으로 보입니다.  
Unet은 과도하게 훈련되기 매우 쉬운데, 만약 결과물이 이상하게 보인다면 과도하게 훈련되었거나 너무 적게 훈련되었을 가능성이 높습니다. 적절한 결과물을 얻는 범위는 좁지만, "좋은 영역(good zone)"은 세트마다 다양하며, 판단하기 어렵습니다.  
만약 생성물이 이상하게 보인다면 문제 해결 팁을 확인하세요.

표준 값은 1e-4입니다. 일반적으로 Unet 값을 건드리지 않는 것이 좋습니다. 다음과 같은 경우에만 Unet 값을 변경하거나:

- 모델이 과적합되는 것처럼 보인다면 Unet을 지나치게 강하게 훈련한 것일 수 있습니다. 학습률을 줄이거나 에포크를 줄이거나 알파를 감소시키거나 기타 감쇠 장치를 사용하여 이를 해결할 수 있습니다.
- 모델이 시각적 노이즈 덩어리를 출력한다면 (일부분이 아니라 문자 그대로 의미없는 노이즈) 학습률을 너무 높게 설정한 것입니다. 최소한 8로 나누어 줄이세요. 아마도 숫자 하나를 빼먹었을 가능성이 높습니다.
- 모델이 너무 약해보이거나 세부 사항을 복제하지 못한다면 학습률이 너무 낮거나 더 많은 스텝이 필요할 수 있습니다.

Unet은 아마도 전체 프로세스에서 가장 복잡한 부분 중 하나입니다. 이것은 이미지의 "계획(plan)"과 색상 및 질감에 관한 정보를 포함하고 있습니다.  
이것을 어떻게 이해하면 좋을까요? 어떤 종류의 점진적인 디테일링 과정으로 생각해보세요. 먼저 행성을 얻게 되고, 그 안에는 대륙, 국가, 그리고 주 또는 지방, 그리고 도시, 거리 등이 있습니다. 이 과정은 "흐릿한 실루엣"이라고 설명될 수 있는 것으로 시작되며 (생성 시작 시 볼 수 있는 것과 유사), 계속해서 "전진"하면서 더 많은 디테일이 추가됩니다. 사람의 경우, 일반적인 자세의 기본부터 시작하여 픽셀이 될 때까지 "확대"되는 과정입니다. 이것이 낮은 해상도가 일부 디테일을 흔들리게 만드는 이유입니다. 더 많은 픽셀을 요청할수록 작업 공간이 많아지기 때문입니다. 이것이 고해상도가 필요한 이유이며, 얼굴을 보정하거나 재구성하는 것이 더 나아지게 만드는 이유입니다.  
**!!! 정보** 이것은 "시각적" 단순화이며, 전체적으로는 수학의 혼합물이며 확률 크루톤을 포함하며 이미지라는 엄격한 의미에서 이미지를 포함하지 않습니다.  
Unet은 여러 블록으로 구성되어 있으며, 이 블록들은 디테일 레벨과 관련이 있습니다. IN 블록은 계획을 결정하고, OUT 블록은 질감과 같은 세부 사항을 결정합니다.

그러므로 Unet을 과도하게 훈련하면 물체를 어디에 배치하거나 어떻게 세부 사항을 처리해야 하는지에 대한 문제가 발생할 수 있습니다. "_눈의 확률, 0.9? 물론 눈을 여기에! 그리고 여기에! 그리고 여기에도!_"와 같이 모든 값이 너무 높아져서 **모든 곳에 모든 것을 놓으려고 시도**하면 디지털 청크 형태의 결과물이 나오게 됩니다.

###   

### 옵티마이저

옵티마이저는 학습 과정을 감독하는 디지털 마술의 일종입니다.

|옵티마이저|필수 인수|노트|VRAM|
|---|---|---|---|
|AdamW8bit|없음|기본값. 지금까지 가장 잘 테스트된 옵티마이저입니다.|낮음|
|AdamW|없음|기본값이지만 32비트입니다. VRAM을 두 배 사용하지만 이론적으로는 더 정밀하며 실제로는 약간 더 나은 결과를 얻을 수 있습니다.|중간|
|DAdaptAdam|--optimizer_args "decouple=True" "weight_decay=0.01" "betas=0.9,0.999" 학습률은 1.0에서 0.1 사이여야 합니다. Constant 스케줄러는 좋은 기본값입니다. 알파 값이 1인 경우 최상의 결과가 나옵니다.|학습률을 동적으로 조정합니다 (적응형). **작업을 위해 인수가 필요하며, 고품질**입니다. 높은 VRAM 사용량. (6.0GB 최소, 6GB 그래픽 카드에서 다른 것을 사용하지 않을 경우 사용 가능)|높음|
|SGDNesterov8bit|없음|동작하며 낮은 VRAM 사용량 (AdamW8bit과 동등). 매우 느린 학습, 2000 스텝은 충분하지 않음. 실제로 더 "최적화된" 것은 아님, 시간이 훨씬 더 많이 필요합니다. 추가 테스트가 필요합니다.|낮음|
|SGDNesterov|없음|SGDNesterov8bit와 동일한 문제, 그러나 AdamW와 같은 더 높은 정밀도입니다.|중간|
|AdaFactor|--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True" 적응형 (기본값)|스케줄러를 무시합니다. Nesterov와 유사한 결과를 제공하지만 적응형이며 VRAM 사용량이 매우 낮습니다. 더 실험해 보기에 유용할 수 있습니다. DAdaptation보다 훨씬 좋아 보입니다.|매우 낮음!|
|Lion|없음|나쁘지 않지만 매우 이상한 결과를 얻을 수 있습니다. 더 높은 정밀도 때문에 다소 무거울 수 있습니다.|중간-높음|
|Lion8bit|없음?|Lion의 새로운 8비트 변형입니다. 낮은 VRAM 설정에서 더 유용해질 수 있습니다.|미정|
|Prodigy|--optimizer_args "decouple=True" "weight_decay=0.01" "d_coef=2" "use_bias_correction=True" "safeguard_warmup=True", d_coef는 다른 유형의 LoRA에 대해 조정이 필요할 수 있습니다.|DAdaptation의 직접적인 업그레이드로, SDXL 및 LyCORIS뿐만 아니라 일반 LoRA에도 적합합니다. 좋은 결과를 제공합니다.|높음|

###   

### 옵티마이저

옵티마이저는 학습에 자체적인 규칙을 제공하며 추측 작업을 단순화하거나 더 적은 리소스를 사용하거나 더 많은 리소스를 사용하여 더 나은 결과를 얻을 수 있도록 도와줍니다. 그들은 나쁜 학습 세트를 유용한 것으로 변환할 수는 없겠지만, 과정을 최적화하는 데 도움을 줄 수 있습니다. 어떤 옵티마이저는 다른 옵티마이저보다 더 잘 작동하므로 몇 가지를 시도해 보는 것이 좋습니다.  
현재 가장 좋은 옵티마이저는 _adaptive_(적응형) 옵티마이저입니다. 이들은 학습률을 손실에 기반하여 자동으로 조정하며, 매 _역전파_ (학습이 모델의 수학적인 혼합물에 변경 사항을 반영할 때)마다 학습률의 균형을 맞춥니다.  
기술적으로 더 복잡하지만 표에서 올바른 옵티마이저 인수를 설정하기만 하면 충분하며, 그런 다음 이미지와 태그만 신경 쓰면 됩니다.  
현재 DAdaptAdam, DAdaptLion 및 Prodigy가 추천되는 옵티마이저입니다. 추측 작업을 모두 제거하고 손상을 피하기 위해 입증된 최적화 과정입니다. Prodigy는 SDXL에서도 작동한다고 보고되므로 익숙해지는 것이 좋습니다.

####   

#### Lion

Lion은 하얀 머리카락을 가진 캐릭터의 머리카락이 무지개색 혼란으로 나타나는 경우와 같이 매우 이상한 결과를 내기도 합니다. 그러나 어쨌든 일부 사람들은 이 옵티마이저를 신뢰합니다. 그래서 제가 모르는 무언가가 더 있을 수도 있습니다. Lion은 모델에 _고유한 풍미를 추가합니다_, 그러나 저는 아직 왜 그런지 설명할 수 없습니다.  
비교적 빠르게 학습하며 몇 가지 경우에 유용할 수 있습니다. 결과물이 어떻게 나오는지 확인하려면 직접 시도해 보는 것이 좋습니다.

####   

#### AdaFactor

다른 한편으로 AdaFactor는 망가지지 않은 결과를 출력했지만 학습이 실제로 약해 보이며 훈련에 조금 더 많은 시간이 필요할 수 있습니다.  
출력에 더 작은 영향을 주기 때문에 **스타일이나 컨셉 학습에 더 적합할 수** 있습니다. 하지만 아직 테스트하지 않았습니다.  
스텝당 시간이 상당히 느리다는 보고도 있습니다.  
TODO: 더 시도해 보기.

####   

#### DAdaptation 🏵️

**!!! 경고** DAdaptation은 작동하려면 특정 인수가 필요합니다. 스케줄러는 constant로 설정되어야 하며 --optimizer_args "decouple=True" "weight_decay=0.01" "betas=0.9,0.999"가 필요합니다.  
**!!! 경고** DAdaptation은 Unet/TE 학습률을 따로 설정하는 데 어려움을 겪는 것 같습니다. Unet=1.0 및 TE=0.5로 설정하는 것이 좋다는 이야기가 있지만 실제로는 양쪽을 모두 1.0으로 설정하고 정렬하도록 하는 것이 가장 좋아 보입니다. (역자 주: 댓글에서 Dadaptation 3.0부터는 unet-te 학습률을 각각 따로 설정 못한다는듯?)  
**!!! 경고** 202306: DAdaptation은 이제 별도로 설치해야 합니다. 가상 환경이 활성화된 상태에서 pip install -U dadaptation로 설치할 수 있습니다.  
**!!! 정보** DAdaptation은 DAdaptAdam으로 이름이 변경되었습니다. 유의해 주세요.

DAdaptation은 _적응형_ 옵티마이저로, 학습 값을 실시간으로 조정하여 직접 제어할 필요가 없게 합니다. 학습 세트가 불량이 아닌 한 (하지만 학습에 모든 것이 해당됩니다) 최소한의 노력으로 매우 좋은 결과를 제공합니다. 현재 시점에서 저는 **사용 가능한 것 중 가장 좋은 옵티마이저**라고 생각합니다.

하지만 큰 주의가 필요합니다. DAdaptation은 상당히 무거워요. **배치 크기 1 (512x512)에서 6.1GB VRAM을 사용하므로 6GB VRAM 사용자는 사용할 수 없습니다**. 이 경우 AdaFactor가 대안이 될 수 있습니다.

**!!! 노트** 옵티마이저의 비결정성 때문에 Alpha 효과를 신뢰할 수 없을 수 있습니다. 이를 염두에 두고 사용하세요.  
다양한 Alpha 설정으로 DAdaptation을 훈련해 보았습니다. Alpha 1이 가장 좋은 결과를 제공하는 것으로 나타났습니다. Alpha 64 (랭크 128과 함께)도 괜찮았습니다.  
반면에 Alpha 0 (= Net Dim)은 비교적 나쁜 결과를 제공했습니다. 더 정확한 숫자를 얻을 때까지 Alpha를 네트워크 크기/랭크의 절반 사이 (Net Dim이 128인 경우 1-64, Net Dim이 32인 경우 1-16)로 유지하는 것을 권장합니다.

기타 옵션:

- 대략 0.1까지의 노이즈 오프셋은 문제를 일으키지 않는 것으로 보입니다.
- Flip_aug도 괜찮지만 비대칭 캐릭터 디자인과 같은 문제가 발생할 수 있습니다.

####   

#### Prodigy

**!!! 경고** Prodigy는 이제 별도로 설치해야 합니다. **가상 환경이 활성화된 상태에서** pip install -U prodigyopt로 설치할 수 있습니다. **kohya-ss와 함께 사용하는 경우에는 필요하지 않습니다. kohya-ss가 스스로 처리합니다.**  
**!!! 경고** 다음 옵티마이저 인수를 사용하여 작동하도록 설정하세요: --optimizer_args "decouple=True" "weight_decay=0.01" "d_coef=2" "use_bias_correction=True" "safeguard_warmup=False" "betas=0.9,0.999"  
Prodigy는 DAdaptation을 직접 업그레이드한 것으로 보이며 많은 외관적 특성을 공유합니다. 동일한 팀에 의해 만들어졌으므로 후속작으로 간주될 수 있습니다.

DAdaptation과 마찬가지로 _적응형_이며 학습을 진행함에 따라 학습률을 조정합니다. 조정이 더 규칙적인 것처럼 보입니다 (**리핏보다는 더 많은 에포크 사용!**). Prodigy는 **SDXL LoRA 훈련 및 LyCORIS 훈련**에도 사용할 수 있으며 높은 성공률을 보였다고 읽었습니다.

저는 Prodigy를 약간만 시험해 보았지만, 정확한 매개변수와 훈련 세트로 성공적인 DAdaptAdam LoRA와 동일한 결과가 눈에 띄게 더 나왔습니다. 특정 테스트는 학습하기 어려운 대상을 사용했기 때문에 더욱 주목 받았습니다. 이미 DAdaptation의 팬이었으므로 직접 업그레이드는 큰 문제였습니다!

DAdaptation의 모든 특성과 주의 사항이 Prodigy에도 있습니다. VRAM 사용량과 속도는 거의 동일하며 설정도 매우 유사하며 옵티마이저 인수를 조금 변경해야 합니다 (이 섹션의 두 번째 경고 또는 위의 표 참조).

####   

#### d

d는 Prodigy가 동적 학습률로 사용하는 값을 의미합니다. TensorFlow에서 보면 lr/d*lr 차트로 표시됩니다.

####   

#### d_coef

범위: 0.5-10 (권장 2)  
동적 학습률 계수입니다. Prodigy가 사용하는 동적 학습률 d의 값을 곱합니다. 이는 학습률을 너무 낮게 결정하거나 **낮은 랭크**를 사용하는 경우 학습률을 스케일 업하는 가장 간단한 방법입니다. Unet/TE 학습률의 값을 2 정도로 늘릴 수도 있지만, 이 방법이 더 안정적인 것 같습니다.  
**권장 값은 0.5에서 2 사이**입니다. 그러나 매우 낮은 랭크 (LyCORIS-LoKR, LyCORIS-(IA)^3, LoRA-LierLa 랭크 <= 8)에서는 최대 10까지 올릴 수 있습니다. 이를 넘어서면 너무 많아져 사용하지 않는 것이 좋습니다. 이는 제 실험 결과와 일치하는 것 같습니다.  
LyCORIS-(IA)^3의 경우 학습률이 거대하게 필요하므로 8-10으로 늘릴 것을 권장합니다.

####   

#### 스케줄러

Prodigy는 constant 스케줄러와 함께 잘 작동합니다. d 값 사용으로 인해 해당 값이 적절하게 상승하거나 하강합니다. scale_weight_norms와 같은 다른 안전장치는 잠재적인 과적합을 방지합니다.  
Prodigy는 constant_with_warmup 스케줄러와도 잘 작동하지만 더 많은 스텝이 필요합니다. 약 10(%) 워머 업에서 보상을 위해 추가적인 에포크 하나 또는 두 개가 필요할 수 있습니다. 워머 업을 사용하는 경우 safeguard_warmup=True를 활성화하는 것이 좋습니다.  
그런 다음 cosine 및 cosine with restarts 스케줄러도 학습률을 상응하는 방식으로 조정하지만, 필요한 경우가 매우 드뭅니다.

####   

#### 최소 SNR 감마

min_snr_gamma의 효과는 Prodigy를 사용할 때 더 흥미롭게 보입니다. 이것은 동적 학습률을 일종의 곱셈 스케일로 보이게 만드는 것 같습니다.  
낮은 값은 낮은 손실을 가져옵니다 (따라서 더 공격적으로 학습함), 반면 높은 값은 높은 손실을 가져옵니다 (학습 세트와 덜 유사함).  
이게 무슨 뜻인가요? **min_snr_gamma <= 5는 캐릭터 학습에 더 적합**하게 만듭니다. [캐릭터 난이도](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1) 차트를 기반으로하면 더 공격적인 학습이 필요한 높은 (C-E) 범주의 캐릭터에서 도움이 될 것이며, 이로 인해 LyCORIS-LoKR에서 더 복잡한 캐릭터를 사용하거나 LoRA-LierLa에서 rank <=16을 사용할 수 있습니다. 값 1은 꽤 강한 것 같았습니다.  
반면에 **min_snr_gamma >= 5는 학습을 더 완만하게 조절**하고 따라서 스타일 및 컨셉에 유리한 **낮은 학습률로 훈련**을 허용합니다.

###   

### 스케줄러

**!!! 노트** 작성 중입니다.  
현재 cosine_with_restarts가 모든 스케줄러 중에서 가장 효과적인 것으로 보입니다. 기본값으로 간주할 수 있습니다.  
스케줄러는 주어진 패턴에 따라 학습률을 변경합니다. 예를 들어 cosine는 학습률을 위아래로 오르내리는 코사인 파형으로 만듭니다.  
그러나 특별한 상황에서는 다른 스케줄러가 선호될 수 있습니다. 예를 들어, 옵티마이저로 DAdaptation을 사용하는 경우 constant로 설정해야 합니다.

|스케줄러|효과|
|---|---|
|constant|학습률이 변경되지 않습니다.|
|constant_with_warmup|상수와 유사하지만 0에서 시작하여 워머 업 스텝 동안 주어진 값에 도달할 때까지 선형적으로 증가합니다.|
|linear|끊임없이 하강하여 끝에서 0이 됩니다.|
|cosine|학습률이 코사인 파형을 따라 위아래로 오르내립니다.|
|cosine_with_restarts|코사인과 유사하지만 일정한 간격에서 완전한 LR에서 시작합니다.|
|polynomial|선형과 유사하지만 더 화려한 곡선으로 진행됩니다.|

숫자는 절대적인 값이 아니기 때문에 수학적으로 정확한 숫자는 없습니다.

이것은 중요하고 또 흥미로운 점입니다. _Unet 또는 TE를 소모하지 않으면서 더 많은 스텝을 학습에 적합하게 넣을수록 좋습니다_. 때때로 학습을 더 많이 하기 위해 학습 속도를 낮추는 것이 필요할 수 있습니다. 물론 이는 더 많은 학습 시간을 필요로하지만 결과가 이미지를 더욱 효과적으로 활용하게 만들어줍니다. 더 높은 학습 속도로 학습 속도를 보상할 수도 있지만 그 효과는 좋지 않을 수 있습니다.  
가장 간단한 방법은 Alpha를 사용하는 것입니다. 이것은 다른 영향을 미치지 않습니다.

학습을 의도적으로 크게 둔하게 설정하는 것은 권장하지 않습니다. 빠르게 효과가 줄어들기 때문입니다. 그러나 스타일이나 컨셉을 학습하는 경우에는 이것이 적용될 수 있습니다.

###   

### 네트워크 차원 (network dimensions)/랭크

**!!! 정보** 네트워크 차원은 "랭크"로도 참조될 수 있습니다. 만약 랭크 128에 대해 읽었다면 네트워크 차원이 128일 것입니다.  

**!!! 경고** 현재 최대 값은 768인 것으로 보입니다. 이를 초과하는 값은 문제를 일으킬 수 있습니다.  

**!!! 경고** 낮은 랭크와 동일한 디테일을 달성하기 위해서는 **보다 높은 학습률**이 필요합니다. **적응형 옵티마이저**가 이 부분을 대신 처리해 줄 것이지만, **아직 수동 학습률을 사용하고 있다면 이를 염두에 두세요**.  
네트워크 차원 또는 랭크는 Unet/TE의 파라미터 수를 나타냅니다. 기본값은 4이지만, 대부분의 캐릭터나 스타일에 대해 64는 아주 좋은 값입니다.  

신비한 닌자 스크롤은 값이 클수록 모델의 "표현력"이 높아지지만, 파일 크기가 더 크다는 단점이 있습니다. 대략 값 x 1.3MB (전체 정밀도인 경우 x 2) 정도입니다. 가령 현재 안전한 최대 크기인 768인 가상 모델은 대략 1GB일 것입니다. 이를 실수로 저장하면 2배가 됩니다.

캐릭터의 경우, 랭크 64가 충분한 경우가 많습니다. 캐릭터가 더 "일반적"이라면 더 낮은 랭크도 가능합니다. 얼굴 특징, 머리카락 스타일 및 옷이 주된 특성인 애니메이션 소녀의 경우 랭크 32 또는 16도 괜찮습니다. 더 독특하거나 복잡한 캐릭터는 64, 92 또는 128에서 더 나아집니다. **현재 방법에서는 92가 감소 효과가 시작되는 지점**으로 보이며, **128은 이미 약간 낭비**입니다. 요즘은 포켓몬 및 만화 캐릭터도 64로 아주 잘 처리할 수 있습니다.

스타일은 더 큰 랭크를 사용할 수 있습니다. **이미지 수**와 최종 값 간에 약간의 상관관계가 있습니다. 이미지가 적은 경우 64 또는 92가 충분합니다. 10000개 이상의 이미지인 경우 128을 시도해보세요. 백만 단위의 이미지 세트를 사용할 때까지 256으로 가지 않아도 됩니다. 그러나 아주 적은 이미지와 작은 랭크로 스타일을 학습하는 것도 가능하므로 추측하거나 결정하기 어려운 경우 더 낮은 값을 선택하도록 노력하세요.

컨셉은 약간 낮을 수 있습니다. 간단한 의복, 포즈 및 복잡하지 않은 배경은 값이 8 이하로도 작동할 수 있습니다.

###   

### 네트워크 알파 (network alpha)

**!!! 경고** 알파는 랭크보다 작거나 같아야 합니다.  
네트워크 알파는 학습 브레이크 또는 덤퍼로 작용합니다. 이는 언제나 _랭크의 값과 관련_이 있습니다.  
이것은 저장될 때 가중치 (모델의 실제 데이터)를 스케일하는 데 사용됩니다. 가중치를 (alpha/net dim)로 곱하여 저장하므로 일부 가중치가 0으로 반올림 에러를 일으키지 않도록 방지하기 위해 도입되었습니다.

- 0으로 설정하거나 net dim과 동일한 값을 사용하면 덤퍼닝이 발생하지 않습니다.
- 기본값은 1로, 학습을 상당히 덤프하므로 보상하기 위해 더 많은 스텝이나 **더 높은 학습률**이 필요합니다.
- 최대 값은 **net dim과 동일한 값**입니다. 더 높은 값은 테스트했을 때 출력이 상당히 **과적합**되었습니다.  
    128의 net dim 값을 사용할 때 숫자가 어떻게 되는지 관찰해보세요:

```
Alpha 0 = Alpha 128 = 128/128 = x1
Alpha 1 = 1/128 = x0.0078125
Alpha 64 = 64/128 = x0.5
Alpha 128 = 128/128 = x1
Alpha 256 = 256/128 = x2
```

###   

### 노이즈 오프셋

**!!! 경고** 너무 높게 설정하면 11 이상의 CFG 스케일과 유사한 "딥 프라이" 효과가 발생할 수 있습니다.

노이즈 오프셋은 학습 시 잠재 요인에 무작위 값을 추가하여 이미지의 동적 범위를 늘리는 기능입니다. 이로 인해 학습 속도가 느려지는 대신 이미지의 동적 범위가 증가합니다.  
이 기능은 **완전히 선택 사항**이며 효과를 원하거나 학습을 덤퍼닝으로 사용하려는 경우에만 사용해야 합니다.

효과는 생성 시 CFG 스케일을 높이는 것과 유사하지만 조금 더 미묘합니다. 너무 높게 설정하면 출력물이 "딥 프라이"될 수 있으므로 주의가 필요합니다. 약간 높여도 약간 더 강렬한 색상이 필요한 경우에는 해가 없습니다.  
0.1의 값을 권장합니다. 학습 덤퍼닝을 상쇄하기 위해 1-2개의 스텝을 더 실행하세요.  
효과는 **학습 세트마다 다양**하며 값이 높을 때 딥 프라이되는 경향이 있는 학습 세트와 그렇지 않은 학습 세트가 있습니다. 보통 정규화된 모델이 더 나옵니다.  
조금 시도해 보았지만 일반적인 사용을 권장할 수 없습니다. 그러나 입력이 흐린 색상을 가진 경우나 출력물이 더 높은 CFG 스케일로 더 좋아 보이는 상황과 같이 일부 상황에서 유용할 수 있습니다. 이렇게 하면 웹 UI에서 값을 높이는 대신에 해당 효과를 적용할 수 있습니다.

- 최근에는 0.06의 값을 사용하여 약간 추가하고 있습니다.
- 알고 있는 내용을 제대로 이해하고 학습 세트가 견고하며 더 많은 대조를 원하는 경우 안전하게 값을 높일 수 있습니다.

###   

### 해상도

정확한 효과를 결정하기 위해 자세한 내용이 필요합니다.  
해상도는 하나의 숫자 (예: "512") 또는 width x height 값 (예: "512x758")으로 지정할 수 있습니다.  
더 많은 학습 시간/VRAM의 대가로 세부 품질과 구성을 높일 수 있지만 부작용도 발생할 수 있습니다.  
너무 높게 설정하면 낮은 해상도 이미지의 품질이 감소할 수 있지만 조금 높게 설정하는 것은 괜찮습니다.

- 512는 잘 작동하는 기본값입니다.
- 지금까지 576 (576x576)은 학습 속도와 VRAM 사용량의 대가로 내 베이크를 일관되게 개선시켜왔습니다.

###   

### 증강

증강은 학습 중에 적용되는 간단한 이미지 효과입니다.  
효과는 미묘한 것부터 "모델을 구원할 만한 효과"까지 다양하지만 학습 속도가 느려집니다.  
사용 가능한 옵션은 다음과 같습니다:  
**!!! 경고** flip_aug 이외의 증강이 활성화된 경우 --cache_latents는 작동하지 않습니다.

|증강|값|효과|캐시 레이턴트 비활성화?|
|---|---|---|---|
|flip_aug|None|이미지를 무작위로 수평으로 뒤집습니다. 언제나 유용하지만, 캐릭터가 매우 비대칭적인 경우에는 사용하지 마세요. 예를 들어 Street Fighter III의 그 사람처럼요.|NO|
|color_aug|None|무작위로 색조 변화를 생성합니다. 색상 범위를 강화하고 비슷한 색상 요소를 약간 더 잘 분리할 수 있습니다.|YES|
|crop_aug|None|큰 이미지를 크기 조정하는 대신 여러 부분으로 나눕니다. 테스트가 필요합니다.|YES|
|face_crop_aug_range|"min,max" (예: "2.0,4.0")|얼굴을 확대하려고 시도합니다. 효과가 눈에 잘 띄지 않을 수 있습니다.|YES|

  

###   

### Min. SNR gamma

이것은 훈련 손실의 무작위 피크를 완화하는 새로운 기능으로, "더 부드러운" 학습을 이끌어냅니다. Unet/TE 문제를 줄이는 데 도움이 될 수 있습니다.  
권장 값은 --min_snr_gamma=5이며, 1은 더 강한 효과를, 20은 거의 어떤 효과도 나타내지 않습니다. 20은 최대 값으로 간주되지만, 1-10 범위를 권장합니다.  
속도 손실이나 VRAM 증가 없이 진행되므로 사용해도 문제가 없습니다. 이것은 꽤 단순하고 직관적인 최적화입니다.  
SNR 감마 사용은 DAdaptation 및 아마도 AdaFactor와 일부 문제를 일으킬 수 있지만, 나는 단 한 번의 주목할 만한 품질 하락만 경험했으므로 항상 사용해도 괜찮을 것입니다. Prodigy와는 더 직관적인 상호작용이 있습니다. 더 낮은 SNR 감마 값은 약간의 브레이킹 효과를 도입할 것이며, 더 높은 값에 대해 증가합니다.

###   

### Scale Weight Norms

이것은 가중치에 너무 큰 값이라고 간주되는 값을 축소하는 기능입니다. 이것은 보통의 방식으로 Unet을 손상시킬 수 있는 이상한 봉우리를 방지합니다. 이것은 본질적으로 Unet 및 TE를 과열로부터 보호하며, 학습을 브레이킹하고 훈련 세트 스타일을 올바르게 버립니다.  
**권장 값은 1**이며, 더 낮은 효과를 얻으려면 더 높은 값을 (최대 10까지) 사용하고, 더 높은 효과를 얻으려면 더 낮은 값을 사용할 수 있습니다.  
이는 **둘 이상의 LoRA 모델을 결합하는 등 다른 모델과의 호환성을 높이는 데 사용됩니다**. 스타일이 일부 손실되더라도 일부 효과가 있습니다.

나는 값이 1일 때 캐릭터의 옷이나 포즈를 변경하기가 조금 더 쉬워지는 것 같다는 것을 발견했습니다.

###   

### 시드

랜덤 시드는 반복 가능한 무작위 숫자 시퀀스를 결정하는 것입니다. 이미지 생성할 때와 마찬가지로 시드는 모델에 영향을 미칩니다.

과제에 적합하지 않은 무작위 시드를 찾을 수도 있으나, 이는 드문 경우입니다. 적응형 옵티마이저를 사용하면 이 경우가 더욱 드물어질 것입니다.  
모든 것을 시도하고 여전히 모델이 좋지 않다면, 시드를 변경해 보는 것이 도움이 될 수 있습니다.

대부분의 난수 생성기는 _의사 난수_로, 일반적으로 난수처럼 보이는 수를 출력하지만 시퀀스 내에서 생성됩니다. 이는 프로그래머가 데이터를 단일 시드로 축소하여 맵, 레벨 또는 생성된 이미지를 생성하는 데 사용할 수 있도록 해줍니다. 시드가 요청되지 않는 경우, 보통 컴퓨터의 시계에서 가져오거나 다른 수단으로 무작위화됩니다.

나는 시작부터 고정된 훈련 시드를 사용해 별다른 문제없이 사용해 왔지만, 남에게 도움을 줄 때 두 가지 경우의 문제 시드가 원인이었던 적이 있습니다. 따라서 매우 드문 가능성이긴 하지만, 다른 해결책이 도움이 되지 않는 경우 이 사항을 염두에 두시기 바랍니다.

###   

### Loss

손실은 설정이라고 보기 힘들지만, 언급할 가치가 있는 내용입니다.  
"손실"은 이미지가 훈련에 사용된 이미지와 얼마나 유사한지를 나타냅니다. 손실이 너무 적으면 훈련 이미지와 매우 유사한 결과물이 생성되며, 더 큰 손실은 모델이 추적을 잃고 무엇이든 하기 시작하는 것을 의미합니다. 따라서 더 큰 손실 = 훈련 세트와 덜 유사한 결과물입니다.  
손실은 일반적으로 0.15에서 0.05 범위 내에 있습니다.  
손실 차트를 검토하려면 로그 폴더를 활성화하고 Tensorflow를 사용하여 그래프 형태로 볼 수 있습니다. 이를 Kohya 스크립트와 Kohya_ss 양쪽에서 수행할 수 있습니다.

---

##   

##   

## 모델 테스트와 디버깅

###   

### 테스트의 중요성

AI 이미징의 무작위한 성질로 인해 첫인상은 오해를 불러일으킬 수 있습니다. 모델이 우연히 첫 3개의 테스트 이미지를 훌륭하게 출력하면, 모델을 좋은 것으로 판단하고 처리할 기회를 놓치게 되며, 나중에 문제점을 발견할 수 있습니다. 첫 몇 장의 이미지가 나쁜 경우도 마찬가지입니다.  
따라서 모델을 테스트하고 다양한 프롬프트, 조합 및 기타 추가 기능을 사용하여 모델의 실제 성능을 확인해야 합니다. XY 그래프 또는 약 5장의 이미지 일괄 처리로 프로세스를 빠르게 진행할 수 있습니다.

![[Pasted image 20240214110715.png]]



다양한 시나리오, 포즈, 옷과 표현을 시도해 보세요. 나는 테스트할 때 사용할 프롬프트 목록을 복사하여 보관합니다.

또한 AI는 시간이 지남에 따라 재미있거나 불쾌한 기묘한 특징을 보일 수 있으며, 이는 처음에는 분명하지 않은 어떤 이유 때문에 발생합니다. 이는 일부 삐딱한 또는 빠진 태그로 인한 경우가 대부분입니다. 과적합의 징후도 처음에는 분명하지 않을 수 있습니다.

즉, 당신은 몇 가지 실질적인 숫자가 필요합니다. **첫인상을 맹신하지 마세요**, 이는 매우 무작위적입니다. 태그를 몇 개 변경하거나 부정적인 부분을 변경해 보세요. 내가 프롬프트를 조금 조정하면 처음 보았을 때는 _나쁘다_라고 생각했던 모델도 꽤 좋아진 경우가 있습니다.

###   

### 강도 (생성 시간 기준)

당신의 LORA는 이미지를 생성할 때 **1.0 강도에서 잘 작동해야 합니다**.

- 이미지를 망치지 않기 위해 강도를 낮춰야 한다면, 과적합이나 훈련이 제대로 이루어지지 않은 것입니다.
    - 이에 대한 예외 사항은 매우 충실한 모델이 1.0에서 잘 작동하지만 추가적인 유연성이나 더 인간적인 신체 계획이 필요한 경우가 있으며, 이 경우 0.8 정도로 낮추면 제대로 작동할 수 있습니다.
    - DAdaptAdam 및 Prodigy는 학습률을 자동으로 보정하여 모델이 1.0에서 쉽게 작동하도록 만드는 데 큰 도움이 됩니다.
- 강도를 높여야 한다면, 여전히 더 많은 훈련이나 높은 훈련 속도를 위한 여유 공간이 있다는 의미입니다. 지금의 생성물이 당신에게 얼마나 마음에 드는지에 따라 추가 훈련 여부가 달라집니다.
    - 이것은 물론 CivitAI 등에서 나타나는 새로운 "컨트롤 슬라이더" LoRA 유형을 제외합니다. 이들은 효과의 곱셈자로써 강도를 사용합니다.
- 강도를 **높여야 한다면, 또는 높이지 않을 때 거의 눈에 띄지 않는다면**, 더 많은 훈련이나 더 높은 훈련 속도가 **필요**합니다.

###   

### 다양한 체크포인트에서 모델 테스트

이것은 당연한 것처럼 들릴 수 있지만, 많은 트레이너들이 자신의 모델을 공개하면서 CivitAI 미리보기에서 훌륭하게 작동하는 모델이 실제 설정에서는 매우 나쁘게 작동하는 것을 볼 수 있습니다. **나는 누군가를 특정하고 있는 것은 아니니, 삽을 내려놓으세요!**  
이러한 문제의 주된 원인은 보통 트레이너가 모델을 테스트할 때 훈련에 사용한 체크포인트만으로 테스트하고, 작동이 잘 된다고 판단하면 업로드하기 때문입니다. 내 생각에는 이 방법이 자신의 필요에 의해 훈련하는 경우에는 나쁘지만은 않은 접근 방식이지만, 공개로 업로드할 때는 가장 인기 있는 모델로 테스트하는 것이 좋습니다. 모델에 혼합된 양이 괜찮은 결과를 얻으려면 혼합 강도를 높이거나 활성화 태그의 가중치를 조정해야 할 수 있습니다.

만약 이것이 대단한 결과를 제공하지 않는다고 생각한다면 그냥 마무리하면 되지만, 명확성을 위해 "이 모델은 XXX 체크포인트에서 최고로 작동합니다"와 같은 공지를 넣는 것이 좋습니다.  
만약 다른 체크포인트에서 훈련하거나 인기 있는 믹스와 함께 작동하도록 훈련 매개변수를 변경하면 의도한 체크포인트의 결과를 망칠 수 있기 때문입니다.

###   

### 과적합

![[Pasted image 20240214110739.png]]



모델이 훈련 이미지를 지나치게 공격적으로 재생산하려고 하거나 결과물이 매우 이상하게 나오면 모델이 "과적합"됩니다.  
이는 보통 Unet이 "과열"될 때 발생합니다. 과열된 Unet은 수학 소화불량으로 인해 확률이 너무 높거나 너무 낮게 설정된 것입니다.

과적합은 보통 훈련을 너무 오래하거나 학습률을 너무 높게 설정할 때 발생합니다. 여기서 각 epoch의 상태를 확인하는 방법은 다음과 같습니다:

- 먼저, 매 epoch마다 스냅샷을 저장하도록 하세요.
- 검색 및 대체로 XY 그래프를 생성하세요. 프롬프트를 "LORA"와 같은 키워드로 설정하고, XY 그래프가 그 용어를 당신의 LORA의 활성화 토큰으로 대체하도록 설정하세요(따라서 1.0 강도의 "LORA"가 "<lora:last:1.0>" 또는 "<lora:last-0000001:1.0>"로 대체됩니다. 모델의 파일 이름에 맞게 설정하세요). 그런 다음 이 그래프를 실행하고 모델이 매 단계마다 어떻게 진행되는지 확인하세요. 자연스러운 진행은 "효과 없음"에서 시작하여 캐릭터나 스타일과의 유사성이 증가하는 것입니다.
    - 이 시점에서 epoch 7/10과 같이 어떤 경우에는 완전히 괜찮고 준비됐다는 것을 알 수 있습니다. 다른 것을 할 필요 없이 그냥 이것을 유지하세요.
- 시작부터 항상 과적합이라면, 훈련 속도를 검토하거나 DAdaptAdam, DAdaptLion 또는 Prodigy와 같은 적응형 옵티마이저를 사용하세요. 이러한 옵티마이저는 학습률을 실시간으로 조정합니다.
- 네트워크 차원이 너무 높거나 알파가 넷 차원보다 높은 경우일 수 있습니다 (알파는 랭크보다 높으면 안 됩니다!).
- 그런 경우에도 여전히 과적합되고 있다면, NAI나 기본 SD가 아닌 것을 훈련하고 있는 경우, 이들에서 훈련을 해보세요. 일부 믹스 모델은 훈련에 적합하지 않을 수 있습니다.
- 다른 모든 것이 실패한다면, 이미지가 너무 많거나 이미지가 결과를 저하시키거나 랭크가 이미지의 모든 계산을 제대로 처리하기에는 너무 낮을 수 있습니다.

####   

#### 추가 네트워크 확장판을 사용하는 경우

!!! 참고: sd-webui 1.5부터는 더 이상 이 작업이 필요하지 않습니다. 개별적으로 강도를 지정할 수 있습니다. <lora:mylora:A:B>와 같이 <lora:mylora:1.0:0.5>를 사용하여 Unet을 1.0으로 설정하고 TE를 0.5로 설정합니다.

웹UI 확장 기능을 통해 Unet과 텍스트 인코더 강도를 개별적으로 조절할 수 있습니다. 이를 통해 어떤 컴포넌트가 과/저적합 인지를 확인하고 값들을 실험해 볼 수 있습니다. 변경 사항은 일반적으로 훈련에 필요한 조정을 반영합니다.  
예를 들어, 모델이 Unet에 대해 0.5 강도에서 작동하고 TE에 대해 1.0 강도에서 잘 작동한다면, Unet 학습률을 절반으로 줄이면 됩니다.  
만약 1.0 강도의 Unet과 2.0 강도의 TE에서 잘 작동한다면, Unet은 그대로 두고 TE 학습률을 두 배로 늘리면 됩니다.  
학습률을 수동으로 설정하는 경우 많은 추측 작업을 절약할 수 있습니다.

###   

### 태그 디버깅


![[Pasted image 20240214110751.png]]


이 경우에는 캐릭터의 충분한 다양한 옷을 입은 예시가 충분하지 않았기 때문에 (메디코 클래스의 가능한 일반적인 스킨 또는 Etrian Odyssey IV의 "메가네 메디코"와 같은) 가방에 대한 공간을 채우기 위한 편향이 있으며, 선장 넥과 허리 주변의 가시적인 스트랩이 있는 특정 디자인입니다.  
가방은 _시각적으로 확인 가능한 가방에 모든 이미지를 올바르게 태그하는 것_으로 해결할 수 있습니다. 이 경우에는 자동 태그가 가장 자주 나왔던 "메신저 가방"을 선택했지만 몇 장의 이미지에만 적용했습니다.  
셔츠 로고나 특정 액세서리를 제거하기가 너무 어려울 수 있습니다. 이 경우, 코트 요소 문제는 제거하기가 더 어려우며 일반적으로 캐릭터와 요소를 분리하는 방법을 가르치는 데 더 많은 이미지와 지식을 필요로 합니다 (가능하면 기존 지식을 활용한) 태그가 필요합니다).  
요청 시 다른 것을 생성할 때 해당 요소의 존재 확률을 줄이기 위해 몇 장의 이미지에서 수동 편집을 수행하거나 수행할 수도 있습니다.  
TE 문제와 비교하면 **차이가 섬세**하지만 훈련을 할수록 더 의미가 있습니다. 또한 그런 종류의 문제는 TE 오류처럼 올고 가는 것이 아니라 _정말로 지속적_이기 때문에 주의를 끌 것입니다. 일반적으로 캐릭터의 옷을 완전히 바꾸거나 (드레스만 입은 캐릭터를 후디나 갑옷에) 노출이 있는 경우에 제거하려면 프롬프트를 사용하거나 대체 의복이나 액세서리에 매우 강한 가중치를 적용해야 합니다.

##   

##   

## 희귀 캐릭터/OC 작성 가이드

-> **노력이 필요합니다. 포기하지 마세요** <-  
_작성 중 (과학은 시간과 많은 시행착오가 필요합니다. 기다려 주세요)_  
->
![[Pasted image 20240214110804.png]]


나는 특히 희귀한 캐릭터에 특화되어 있으며, 좋아하는 캐릭터에 대한 새 이미지가 몇 장만 있는 경우 어떻게 대처해야 하는지에 대한 조언을 이 섹션에서 업데이트할 것입니다.  
특정 AI 출력을 더 다듬어진 캐릭터로 변환하려는 사람들의 흥미로운 현상을 관찰했는데, 이는 근본적인 수준에서 나를 매혹시키므로 이 조언을 확장하여 도움을 드릴 계획입니다.

어떤 형태의 이미지 편집기를 준비하십시오 (저는 [Krita](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1)를 추천합니다만, 포토샵, GIMP, [Paint.NET](https://oo.pe/http://Paint.NET "paint.net/ (외부 사이트)") 또는 편한 다른 프로그램이 있으면 사용하셔도 됩니다. 어떤 것도 사용하지 않는 경우에는 Krita(무료이며 실제 아트 작업에 사용하기에 좋습니다)를 다운로드하고 몇 가지 도구에 익숙해지세요.

###   

### 이미지 준비

- 이미지의 품질과 다양성에 따라 최소 10에서 25장의 이미지가 필요합니다. 더 적은 이미지로도 성공한 시도가 있었지만, 이는 운이 결정합니다. 100장의 이미지가 금기금을 측정하는 기준이지만, 여기서는 제약 조건 아래에서 작업하는 것을 이해합니다.  
    - 이미지가 한 두 장만 있다면, **분명히 AI와 함께 작업해야 합니다**().
- 조금 비판적인 시선을 가져야 합니다. 선택할 수 있다면, 더 나은 이미지를 선택하거나 적어도 _당신에게_ 더 나은 이미지를 선택하십시오.  
    - 선택할 수 없다면, 이미지 스타일의 영향을 줄이기 위해 정규화를 사용하십시오.  
    - 필요한 경우 친근한 시선 한 쌍에게 물어보십시오.
- 인간 예술가도 AI와 마찬가지로 항상 정확하지 않고 실수를 저지르며, 이러한 실수가 AI에 전달됩니다. 여러 **시각과 스타일의 동일한 대상**이 필요합니다. 다른 표현으로, 잘못된 헤어 스타일, 눈 색깔, 잘못된 의류 요소와 같은 예술가의 실수는 피하는 것이 좋습니다. 다른 것이 없다면, **편집의 목적은 정확성입니다**.  
    - 때로는 이런 일이 정식 아트 안에서도 발생합니다. 이러한 경우, 가장 좋아하는 변형을 선택하고 동일한 방법으로 강화하려고 노력하세요.  
    - image2image나 inpaint를 사용해서 작동할 수 있지만, 1회에서 무한한 시도까지 걸릴 수 있습니다. 필요한 경우 수동으로 도와주세요.
- 적어도 몇 가지 구도와 다른 의상이 필요합니다. 또는 의상에 여러 개의 레이어가 있는 경우 레이어를 일부 제거한 사진을 토글할 수 있도록 하며, 태그를 이에 맞게 지정하면 됩니다.  
    - 다시 말하지만, 이것이 항상 가능한 것은 아닙니다. 캐릭터가 항상 동일한 의류나 자세로 그려지는 것은 흔한 일이며, 특히 실제로 묘한 경우입니다. 편집하거나 누군가에게 도움을 요청해야 할 것입니다 (그 누군가 [AI일 수 있습니다](https://arca.live/b/aiart/84182288?mode=best&target=all&keyword=lora&p=1)).
- 3D 모델은 가능하지만, 렌더링을 생성하려는 경우가 아니라면 스타일을 완화시키기 위해 정규화를 사용하십시오. 또한 렌더를 변경하려면 렌더를 이미지로 바꾸십시오.
- **기도하세요**

###   

### 이미지 편집

####   

#### Krita 간단한 스타터 팩

전체 자습서는 이 글의 범위를 벗어나므로, 기본 자습서 비디오를 찾아보고 이미지를 Krita에서 sd-webui로 전송하고 받을 수 있도록 플러그인을 설정하는 것이 좋습니다(하지만 복사/붙여넣기도 잘 작동합니다).  
조금 하다가 이 도구들이 꽤 유용하다고 느끼는데 도움이 될 것입니다.


![[Pasted image 20240214110816.png]]



이 네 가지 도구는

모두 브러시 프리셋의 디지털 섹션에 배치되어 있으며, 편집 도구의 기본이 됩니다.

- Distort Move은 커서 아래의 것을 동일한 방향으로 이동시킵니다. 이를 사용하여 짧은 팔다리를 고치거나 머리카락을 길게 또는 짧게 만들고 이미지를 일반적으로 변형할 수 있습니다. 블러되므로 이미지의 해상도가 클수록 더 좋습니다. img2img/inpaint를 사용하여 결과를 고치십시오.
- Adjust Color은 커서 아래 영역의 색상을 선택한 주요 색상으로 변경합니다. 이로서 머리카락/눈/의류 색상을 수정할 수 있습니다. 원하는 결과를 얻으려면 여러 번 수행해야 할 수도 있습니다.
- Blender Blur는 커서 아래의 영역을 흐리게 만듭니다. 자신의 편집을 맞추기 위해 유용하며, 더 작은 실수를 덜 눈에 띄게 만들거나 JPEG 아티팩트를 제거하는 데 사용할 수 있습니다.
- Basic-1은 기본 단색 브러시로, 뭔가를 그리는 데 유용합니다.

개인적으로 이 도구들을 태블릿을 사용하지 않고도 편집하는 데 잘 활용할 수 있다고 확인했습니다. 저도 세부 기술을 필요로 하지 않는 조언을 제공하려고 마우스만 사용했습니다.

기술이 좋지 않다면 편집을 작은 단계로 수행하여 취소 키(컨트롤+z)로 쉽게 되돌릴 수 있도록 하고 처음부터 다시 시작하지 않아도 되도록 하십시오. 레이어를 사용할지 여부는 당신의 재량대로 결정하되, 복사/붙여넣기를 하려면 _모두 선택_하고 _모두 병합 복사_하여 현재 레이어만 복사하지 않도록 해야 합니다(또한 가장자리를 벗어나 그리는 경우 크기를 넘어갈 수 있으므로 항상 모두 선택하고 모두 병합 복사를 해서 항상 같은 크기임을 보장하십시오).

기억하세요, AI는 당신의 편집을 "잘 통합된" 변화로 바꿀 수 있도록 도와줄 수 있습니다.

###   

### AI를 활용하여 도움 받기

**Inpaint와 img2img는 당신의 동반자입니다**. img2img (denoise ~0.7)를 사용하여 공식 아트를 극적으로 변경하여 AI에게 스타일이나 자세의 변화를 가져다줄 수 있습니다. 가능한 높은 해상도로 사용하고, 얼굴과 미세한 세부 사항을 보다 일관성 있게 만들기 위해 얼굴과 미세한 세부 사항을 inpaint (다시 말하지만 _마스크된 부분만_)하세요.  
또한 inpaint를 사용하여 예를 들어 의류를 변경한 다음 일관성을 위해 inpaint (다시 말하지만 _마스크된 부분만_ 및 더 높은 해상도에서)할 수 있습니다.  
"의류 변경"이라고 하면 어떤 의류 부분 위에 약간의 일관성 있는 색 덩어리를 그리거나 지울 수 있으며, 다른 이미지에서 붙여넣거나 스톡 아트, 다른 AI 생성물 등을 위에 올려 붙이면 됩니다. **어떤 방법을 사용하든 상관없습니다**. 목적을 달성하면 됩니다.  
비슷하게, 동일한 원칙을 사용하여 요소를 제거할 수 있습니다. 예를 들어, LoRA가 항상 가방, 끈 또는 작은 방어구 부품과 같은 요소를 생성하고 다른 의류로 퍼지는 경우, 이미지의 사본을 만들어 해당 요소를 제거하고 공백을 채우기 위해 inpaint를 사용하여 시도해보십시오. 마찬가지로 캐릭터가 셔츠 로고나 펜던트와 같은 액세서리를 가지고 있는 경우 또한 다른 이미지의 사본을 만들어 제거하십시오.

또한 ControlNet은 도움을 받는 또 다른 유효한 옵션입니다. 그레이스케일 이미지를 컬러로 칠하거나 빠른 스케치를 더 복잡한 이미지로 바꾸거나 다른 AI 출력을 다양한 스타일이나 이미지로 바꾸는 데 도움이 될 수 있습니다. 또한 스케치를 사용하고 색칠된 이미지를 충분히 확보한 경우 스케치를 넣어보세요. 이렇게 하면 Unet이 작업할 "실루엣"이 더 많이 생성될 것이며 (그리고 일반적으로 부정적인 프롬프트의 "단색"에서는 아마도 폐기될 것입니다).

###   

### AI 출력물을 입력으로 사용하기

**!!! ⚠️경고⚠️** 학습을 위해 AI 이미지를 생성할 때 CFG Scale을 확인하세요. 일반적인 이미지를 생성할 때보다 낮게 설정하여야 합니다. CFG 스케일이 너무 높으면 더 높은 대비와 더 포화된 색상이 LoRA에 피해를 줄 수 있으므로, 상당히 완화된(5-7의 CFG 스케일) 형태로 유지하세요. 물론, 스타일을 위해 포화된 색상을 원하는 경우 자유롭게 사용하세요. 생성 시, 색상은 더 많이 포화될 것입니다.

안정된 확산은 루프백 훈련을 방지하기 위해 투명한 워터마크를 추가하지 **않습니다**. 이와 반대로 주장하는 소문은 무시하세요.

AI 출력물은 AI 입력으로 완전히 유효합니다. 그러나 이들에 대해 _정말 꼼꼼하게_ 검토해야 합니다. **그들은 특정 모델인 OrangeMixes와 같은 모델처럼 보이도혹 결과물에 강력한 영향을 미칠 수 있습니다**.  
이상적으로는 AI 이미지가 교육 해상도와 동일한 크기여야 합니다.

표준 SD 실수는 일반적으로 많은 수의 고품질 데이터로 정규화되지 균형을 이루지 않는 한 결과물에 쉽게 영향을 줄 수 있으며 (_이 글을 읽고 있는 것으로 봐선 그런 여유가 없을 것입니다_), 가능한 한 적절하게 만들어야 합니다. 의심스러운 경우 주변인에게 두 번째 의견을 얻어 보세요 (동일한 대상의 많은 출력을 보면 명백한 오류를 놓치는 경우가 있을 수 있습니다).

손은 일반적으로 포기해야 할 대상입니다만, 적어도 올바른 손가락 수가 있어야 합니다. 필요한 경우 Inpaint를 사용하여 제대로 작동하도록 하세요. 예술 기술이 있다면 스스로 올바르게 수정해야 할 것입니다.

###   

### GACHAMAX 임시 LoRA 방법 (마지막 수단!)

규모가 매우 큰 제약 조건 (예: 하나의 이미지)인 경우, 말하자면 "무차별적인" 과정을 거칠 수 있습니다.  
이는 상대적으로 시간이 많이 소요되며 운이 약간 관련되어 있습니다.

1. 하나의 이미지만 사용하여 간단하고 빠른 모델 (10 에포크, 5분 정도 소요될 것입니다)을 훈련시키고 나온 것 중에서 가능한 최상의 유사성을 유사한 스타일로 생성해 보세요. 필요한 경우 inpaint 또는 img2img를 사용하여 작동하도록 하거나 편집하세요. 처음부터 시작하는 것보다는 쉬울 것입니다.
2. 새로운 이미지로 빠른 모델을 다시 훈련시키세요.
3. 어느 정도의 일관성이 나타날 것이므로, 처음보다 더 쉬워지며 이제 4개의 적절한 이미지를 더 적은 시간 내에 얻을 수 있을 것입니다.
4. 이 시점에서 다른 모델로 전환하거나 스타일 LoRA를 추가하여 스타일의 다양성을 더하고 6개의 이미지 정도를 얻으려고 해 보세요.
5. 또 다른 빠른 모델을 훈련시키세요. 이제 조금 더 긴 시간이 걸릴 수 있지만 10 에포크에서 여전히 매우 빠를 것입니다.
6. 이제 이미지 중 눈에 띄는 것이 있는지 확인하고 필요한 경우 제거하고 최상의 이미지로 대체하세요.
7. 가능한 많은 좋은 이미지를 얻으려고 하고 그 과정을 반복하여 결과물을 견고하고 유연하게 만들어 보세요.

이것은 매우 운에 따라 달라질 수 있는 접근 방식이며 시간을 절약하기 위해 편집이 여전히 필요할 수 있지만, 결국 _어떤 경우에는_ 작동할 것입니다.

###   

### 일반적인 조언

어려운 캐릭터로부터 정말 좋은 모델을 얻기 위해서는 노력이 필요합니다. 캐릭터의 복잡성에 따라서 _좋은_ 결과를 얻으려면 최소한 5번 이상 시도해야 할 수 있습니다 (물론 나의 정확성과 유연성 기준에 따라, 물론 두 번째 시도나 첫 번째 시도가 충분하다면 그냥 마무리하고 완료라고 하세요).

순전히 AI 생성 이미지만을 사용하는 경우, 스타일 유연성을 더 갖기 위해 하나의 체크포인트에 고수하지 않으려고 하세요.

###   

### 예술 기술을 가진 독자를 위한 팁

일부 예술 기술이 있다면, 자신의 캐릭터와 스타일을 훈련시킬 수 있습니다.  
마찬가지로 기존 캐릭터를 훈련시키는 경우, 공개 이미지와 팬아트로는 불가능한 범위에서 모델을 더욱 자세히 사용자 정의할 수 있습니다. 더 많은 각도, 자세, 의류, 표현 및 비율을 제공하여 모델의 범위를 확장할 수 있습니다. 모든 훈련된 모델은 선별과 표준 때문에 다릅니다만, 자신의 아트워크를 제공하면 그로 인해 훨씬 더 독특해질 수 있습니다.

나는 내 이미지와 게으른 스케치에도 훈련이 꽤 긍정적으로 반응한 적이 있으며, 나는 아트에서 최고는 아니지만 매우 만족스럽게 활용했습니다. 이 주제에 더 많은 실험을 위한 더 많은 자유 시간이 있을 때, 이 섹션을 더 확장하겠습니다.

---

## SAMPLE POWERSHELL SCRIPT (WINDOWS)

**START COPYING BELOW HERE, SAVE TO A TEXT FILE, EDIT PATHS AND OPTIONS AS NEEDED**

```
# Config:
$ckpt = "X:\SD\voldy\models\Stable-diffusion\animefull-final-pruned.ckpt"; #Full path to model you want to train FROM, or base model.
$image_dir = "X:\SD\training\my_concepts_folder"; #Data set folder
$output = "X:\SD\voldy\models\lora"; #Output folder for your baked LORAs.
$reg_dir = "X:\SD\training\reg"; #Only use these for dreambooth style training. Point to an empty folder otherwise.

$train_batch_size           = 1        #Amount of images to process at once. I have 8GB of VRAM so I left it at 1, it just worked. Raise if you got more VRAM.
$learning_rate              = 0.0001   #Unet learning rate.
$text_encoder_learning_rate = 0.00005  #Text Encoder learning rate. This is the recommended value.
$num_epochs                 = 8        #Total number of epochs (amount of times the entire set is repeated)
$save_every_n_epochs        = 1        #Save checkpoints every X epochs.
$resolution                 = 512      #Resolution to work at. Higher requires more training for the unet and more VRAM.
$network_dim                = 128      #AKA Rank. Higher for more resemblance to the training images and bigger file size. 96-192 for characters. 160 was good for me.
$network_alpha              = 128      #Must be equal or lower than network dim. Dampens learning the lower it is, but avoids rounding issues.
$noise_offset               = 0.0      #Increases dynamic range of outputs. Every 0.1 dampens learning quite a bit, do more steps or higher training rates to compensate.
$clip_skip                  = 2        #Set it to 2 if you train from NAI.
$optimizer                  = "AdamW8bit" # Valid values: "AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SDGNesterov8bit", "DAdaptation", "AdaFactor"
# Default AdamW8bit (old --use_8bit_adam). DAdaptation requires setting learning rates to values between 0.1 and 1.0 as it tweaks them during training.
$scheduler                  = "cosine_with_restarts"

# End of config

$learning_rate              = $learning_rate * $train_batch_size # Seems to work better for the Unet.

.\venv\Scripts\activate #Activate python venv before starting.

accelerate launch --num_cpu_threads_per_process 8 train_network.py `
    --network_module="networks.lora" `
    --pretrained_model_name_or_path=$ckpt --train_data_dir=$image_dir --reg_data_dir=$reg_dir --output_dir=$output `
    --caption_extension=".txt" --shuffle_caption --keep_tokens=1`
    --prior_loss_weight=1 `
    --resolution="$resolution" `
    --enable_bucket --min_bucket_reso=320 --max_bucket_reso=960 `
    --train_batch_size="$train_batch_size" `
    --learning_rate="$learning_rate" --unet_lr="$learning_rate" --text_encoder_lr=$text_encoder_learning_rate `
    --max_train_epochs=$num_epochs `
    --mixed_precision="fp16" --save_precision="fp16" `
    --optimizer_type="$optimizer" --xformers `
    --save_every_n_epochs="$save_every_n_epochs" `
    --save_model_as=safetensors `
    --clip_skip="$clip_skip" `
    --seed=420 `
        --flip_aug `
    --network_dim="$network_dim" --network_alpha="$network_alpha" `
    --max_token_length=225 `
    --cache_latents `
    --lr_scheduler="$scheduler" `
        --noise_offset="$noise_offset"
```

##   

## SAMPLE BASH SCRIPT (LINUX)

**START COPYING BELOW HERE, SAVE TO A TEXT FILE, GIVE IT EXECUTABLE PERMISSIONS, ETC**



```
#!/usr/bin/env bash
#Set these paths as required:
WEBUI_DIR="/mnt/DATA/AI/stable-diffusion-webui/" # SD-webui folder
MODEL_DIR="/mnt/DATA/AI/stable-diffusion-models/Stable-diffusion/" # Model folder
TRAIN_DIR="/mnt/DATA/AI/sd-scripts" # Kohya scripts folder

function training_Kohya_lora {
    cd "$TRAIN_DIR" || {
        echo "Folder not found or not accessible."; exit 1
    }

    local training_set="penny" #Set this to the base folder for a character.
    local ckpt="$MODEL_DIR/animefinal-full-pruned.ckpt" # Base model(checkpoint) to finetune
    local image_dir="$TRAIN_DIR/SETS/$training_set/"
    local reg_dir="$TRAIN_DIR/SETS/$training_set/reg/" #Regulation image folder. Optional, you can point it to an empty folder if you don't want them.
    local output="/mnt/DATA/AI/stable-diffusion-models/LORA/" #Folder to save outputs. WARNING: Will overwrite existing files.

    local learning_rate="0.0001" #Learning rate. Remember this is supposed to be a magnitude larger than a dreambooth equivalent. Worked well for me at this rate.
    local text_encoder_lr="0.00005" #Learning rate for TEXT ENCODER. This is the value suggested in the ninja scrolls. Seems to work better for details.
    local train_batch_size="2" #Amount of images to process at once. I have 8GB of VRAM so I left it at 1, it just worked. Raise if you got more VRAM.
    local num_epochs="6" #Total number of epochs (amount of times the entire set is repeated)
    local save_every_x_epochs="2" #Save checkpoints every X epochs.
    local network_dim="160" #Higher for more resemblance to the training images and bigger file size. 96-192 for characters.
    local scheduler="cosine_with_restarts"

    . venv/bin/activate #Activate your venv before starting.

    accelerate launch --num_cpu_threads_per_process 8 train_network.py \
    --network_module="networks.lora" \
    --pretrained_model_name_or_path="$ckpt" --train_data_dir="$image_dir" --reg_data_dir="$reg_dir" --output_dir="$output" \
    --output_name="${training_set}_last_e${num_epochs}_n${network_dim}" \
    --caption_extension=".txt" --shuffle_caption \
    --prior_loss_weight=1 \
    --network_alpha="$network_dim" \
    --resolution=512 \
    --enable_bucket --min_bucket_reso=320 --max_bucket_reso=768 \
    --train_batch_size="$train_batch_size" \
    --gradient_accumulation_steps=1 \
    --learning_rate="$learning_rate" --unet_lr="$learning_rate" --text_encoder_lr="$text_encoder_lr" \
    --max_train_epochs="$num_epochs" \
    --mixed_precision="fp16" --save_precision="fp16" \
    --use_8bit_adam --xformers \
    --save_every_n_epochs="$save_every_x_epochs" \
    --save_model_as=safetensors \
    --clip_skip=2 \
    --seed=420 \
    --flip_aug --color_aug --face_crop_aug_range="2.0,4.0" \
    --network_dim="$network_dim" \
    --max_token_length=150 \
    --lr_scheduler="$scheduler" \
    --training_comment="LORA:$training_set"
}

training_Kohya_lora "$@"

```



원문 출처: [https://rentry.org/59xed3](https://oo.pe/https://rentry.org/59xed3 "rentry.org/59xed3 (외부 사이트)")
끝! 보시느라 수고하셨습니다!
