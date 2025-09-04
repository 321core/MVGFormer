# CMU Panoptic to Mixamo Converter

이 스크립트는 CMU Panoptic 데이터셋의 hdPose3d_stage1_coco19 JSON 파일들을 Mixamo 호환 BVH 포맷으로 변환합니다.

## 개요

- **입력**: CMU Panoptic JSON 파일 (body3DScene_*.json)
- **출력**: Mixamo 호환 BVH 애니메이션 파일
- **변환**: 3D 위치 좌표 → 관절 회전 기반 스켈레톤 애니메이션

## 사용법

### 기본 사용법
```bash
python convert_to_mixamo.py <입력_디렉토리> <출력_디렉토리>
```

### 옵션
- `--person_id`: 추출할 사람 ID (기본값: 0)
- `--sequence_name`: 시퀀스 이름 (자동 감지됨)

### 사용 예시

1. **단일 시퀀스 변환**
```bash
python convert_to_mixamo.py data/panoptic/160906_pizza1/hdPose3d_stage1_coco19 ./output
```

2. **특정 사람 ID 지정**
```bash
python convert_to_mixamo.py data/panoptic/160906_pizza1/hdPose3d_stage1_coco19 ./output --person_id 1
```

3. **커스텀 시퀀스 이름**
```bash
python convert_to_mixamo.py data/panoptic/160906_pizza1/hdPose3d_stage1_coco19 ./output --sequence_name my_animation
```

## 출력 파일

변환된 BVH 파일은 다음 형식으로 저장됩니다:
- `{시퀀스이름}_person{ID}.bvh`
- 예: `160906_pizza1_person0.bvh`

## 지원하는 관절

### CMU Panoptic (15개 관절)
0. neck
1. nose  
2. mid-hip
3. l-shoulder
4. l-elbow
5. l-wrist
6. l-hip
7. l-knee
8. l-ankle
9. r-shoulder
10. r-elbow
11. r-wrist
12. r-hip
13. r-knee
14. r-ankle

### Mixamo 스켈레톤
- Hips (루트)
- Spine, Spine1, Spine2
- Neck, Head
- LeftShoulder, LeftArm, LeftForeArm, LeftHand
- RightShoulder, RightArm, RightForeArm, RightHand
- LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase
- RightUpLeg, RightLeg, RightFoot, RightToeBase

## 배치 처리 스크립트

여러 시퀀스를 일괄 처리하려면:

```bash
# 모든 시퀀스 변환
for seq_dir in data/panoptic/*/hdPose3d_stage1_coco19; do
    seq_name=$(basename $(dirname "$seq_dir"))
    echo "Converting $seq_name..."
    python convert_to_mixamo.py "$seq_dir" ./mixamo_output --sequence_name "$seq_name"
done
```

## 테스트 결과

스크립트는 성공적으로 테스트되었습니다:
- 160906_pizza1 시퀀스: 6582 프레임 처리 완료
- 출력: pizza1_test_person0.bvh (정상 생성)

## Mixamo에서 사용하기

1. 생성된 BVH 파일을 Mixamo Studio에 업로드
2. 캐릭터에 적용하여 애니메이션 재생
3. 필요시 추가 편집 및 내보내기

## 주의사항

- 현재 구현은 기본적인 관절 회전 변환을 사용합니다
- 더 정확한 결과를 위해서는 역운동학(IK) 알고리즘 개선이 필요할 수 있습니다
- confidence 값이 0.1 이하인 관절은 무효한 것으로 간주됩니다

## 요구사항

- Python 3.6+
- numpy
- json (내장)
- glob (내장)
- argparse (내장)
