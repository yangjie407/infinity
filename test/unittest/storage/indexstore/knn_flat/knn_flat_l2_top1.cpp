//
// Created by jinhai on 23-9-25.
//


#include <gtest/gtest.h>
#include "base_test.h"
#include "storage/data_block.h"
#include "common/types/info/decimal_info.h"

#include "common/types/info/array_info.h"
#include "common/types/info/embedding_info.h"
#include "main/profiler/base_profiler.h"
#include "main/logger.h"
#include "main/stats/global_resource_usage.h"
#include "main/infinity.h"
#include "storage/indexstore/knn_flat/knn_flat_l2_top1.h"

class KnnFlatL2Top1Test : public BaseTest {
    void
    SetUp() override {
        infinity::GlobalResourceUsage::Init();
        std::shared_ptr<std::string> config_path = nullptr;
        infinity::Infinity::instance().Init(config_path);
    }

    void
    TearDown() override {
        infinity::Infinity::instance().UnInit();
        EXPECT_EQ(infinity::GlobalResourceUsage::GetObjectCount(), 0);
        EXPECT_EQ(infinity::GlobalResourceUsage::GetRawMemoryCount(), 0);
        infinity::GlobalResourceUsage::UnInit();
    }
};

TEST_F(KnnFlatL2Top1Test, test1) {
    using namespace infinity;

    i64 dimension = 4;
    i64 top_k = 4;
    i64 base_embedding_count = 4;
    UniquePtr<f32[]> base_embedding = MakeUnique<f32[]>(sizeof(f32) * dimension * base_embedding_count);
    UniquePtr<f32[]> query_embedding = MakeUnique<f32[]>(sizeof(f32) * dimension);

    {
        base_embedding[0] = 0.1;
        base_embedding[1] = 0.2;
        base_embedding[2] = 0.3;
        base_embedding[3] = 0.4;
    }

    {
        base_embedding[4] = 0.2;
        base_embedding[5] = 0.1;
        base_embedding[6] = 0.3;
        base_embedding[7] = 0.4;
    }

    {
        base_embedding[8] = 0.3;
        base_embedding[9] = 0.2;
        base_embedding[10] = 0.1;
        base_embedding[11] = 0.4;
    }

    {
        base_embedding[12] = 0.4;
        base_embedding[13] = 0.3;
        base_embedding[14] = 0.2;
        base_embedding[15] = 0.1;
    }

    {
        query_embedding[0] = 0.1;
        query_embedding[1] = 0.2;
        query_embedding[2] = 0.3;
        query_embedding[3] = 0.4;
    }

    KnnFlatL2Top1<f32> knn_distance(query_embedding.get(),
                                    1,
                                    dimension,
                                    EmbeddingDataType::kElemFloat);

    knn_distance.Begin();
    knn_distance.Search(base_embedding.get(),base_embedding_count, 0);
    knn_distance.End();

    f32* distance_array = knn_distance.GetDistanceByIdx(0);
    CompoundID* id_array = knn_distance.GetIDByIdx(0);
    EXPECT_FLOAT_EQ(distance_array[0], 0);
    EXPECT_FLOAT_EQ(id_array[0].segment_id_, 0);
    EXPECT_FLOAT_EQ(id_array[0].segment_offset_, 0);

}