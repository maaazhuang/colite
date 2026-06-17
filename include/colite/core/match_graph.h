#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "colite/core/feature.h"
#include "colite/core/two_view_geometry.h"
#include "colite/core/types.h"

namespace colite {

// 图像对键。始终按较小 ID 在前存储，减少 (i,j)/(j,i) 重复。
struct ImagePair {
  ImageId image_id1 = kInvalidImageId;
  ImageId image_id2 = kInvalidImageId;

  static ImagePair Ordered(ImageId a, ImageId b) {
    return a <= b ? ImagePair{a, b} : ImagePair{b, a};
  }

  bool operator<(const ImagePair& other) const {
    return std::tie(image_id1, image_id2) <
           std::tie(other.image_id1, other.image_id2);
  }
};

// 特征与匹配缓存。它不属于最终稀疏模型，避免 Reconstruction 被中间数据污染。
struct MatchGraph {
  std::map<ImageId, FeatureSet> features;
  std::map<ImagePair, std::vector<FeatureMatch>> matches;
  std::map<ImagePair, TwoViewGeometry> verified_geometries;
};

// 图中的一条已验证边。它不拥有几何对象，只作为邻接查询的轻量视图。
struct MatchGraphEdge {
  ImagePair pair;
  const TwoViewGeometry* geometry = nullptr;

  std::size_t NumInliers() const;
  ImageId OtherImageId(ImageId image_id) const;
};

// 写入单张图像的特征缓存。
void SetFeatures(MatchGraph* graph, ImageId image_id, const FeatureSet& features);

// 写入两张图像的原始匹配。若调用方传入反向 image_id，会同步交换 idx1/idx2，
// 保证内部存储始终满足 matches[pair].idx1 属于 pair.image_id1。
void SetMatches(MatchGraph* graph,
                ImageId image_id1,
                ImageId image_id2,
                const std::vector<FeatureMatch>& matches);

// 写入两张图像的几何验证结果。内部同样按 ImagePair 有序键存储。
void SetVerifiedGeometry(MatchGraph* graph,
                         ImageId image_id1,
                         ImageId image_id2,
                         const TwoViewGeometry& geometry);

const FeatureSet* FindFeatures(const MatchGraph& graph, ImageId image_id);
const std::vector<FeatureMatch>* FindMatches(const MatchGraph& graph,
                                             ImageId image_id1,
                                             ImageId image_id2);
const TwoViewGeometry* FindVerifiedGeometry(const MatchGraph& graph,
                                            ImageId image_id1,
                                            ImageId image_id2);

// 查询某张图像所有已验证邻居，并按内点数从多到少排序。
std::vector<MatchGraphEdge> FindVerifiedNeighbors(const MatchGraph& graph,
                                                  ImageId image_id);

}  // namespace colite
