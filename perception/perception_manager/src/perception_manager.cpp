#include "perception_manager/perception_manager.hpp"
#include <thread>

using namespace std::chrono_literals;

PerceptionManager::PerceptionManager() : Node("perception_manager")
{
    RCLCPP_INFO(this->get_logger(), "Loading object configurations from parameter server...");

    // YAML 파일의 파라미터 key-value 쌍을 읽기
    this->declare_parameter("object_names", rclcpp::ParameterValue(std::vector<std::string>()));
    std::vector<std::string> object_names = this->get_parameter("object_names").as_string_array();

    for (size_t i = 0; i < object_names.size(); ++i)
    {
        // === 데이터 구조 개선: object_data_map_을 기본 저장소로 사용 ===
        int object_id = i;
        const auto& object_name = object_names[i];
        
        object_data_map_[object_id].name = object_name;
        object_data_map_[object_id].frame_id = object_name;
        object_id_to_name_map_[object_id] = object_name; // 기존 맵도 유지

        // === 주요 변경점: 파라미터 이름 수정 ===
        std::string param_name = "grasping_offsets." + object_name; // "grasping_offsets." 접두사 추가
        this->declare_parameter(param_name, rclcpp::ParameterValue(std::vector<double>()));
        std::vector<double> pose_vector = this->get_parameter(param_name).as_double_array();

        if (pose_vector.size() == 7)
        {
            geometry_msgs::msg::Pose pose;
            pose.position.x = pose_vector[0];
            pose.position.y = pose_vector[1];
            pose.position.z = pose_vector[2];
            pose.orientation.x = pose_vector[3];
            pose.orientation.y = pose_vector[4];
            pose.orientation.z = pose_vector[5];
            pose.orientation.w = pose_vector[6];
            
            // === 데이터 구조 개선: map에 grasping_offset 직접 저장 ===
            object_data_map_[object_id].grasping_offset = pose;

            RCLCPP_INFO(this->get_logger(), "Loaded grasping offset for %s: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
                object_name.c_str(), pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Failed to load valid grasping_offset for %s. Parameter '%s' should be a vector of 7 doubles.", object_name.c_str(), param_name.c_str());
        }
    }

    // TF 버퍼와 리스너 초기화
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 오브젝트 위치 정보 서비스 생성
    get_object_info_service_ = this->create_service<perception_interfaces::srv::GetObjectInfo>(
        "perception_manager/get_object_info",
        std::bind(&PerceptionManager::get_object_info_callback, this, std::placeholders::_1, std::placeholders::_2));

    ft_sensor_subscriber_ = this->create_subscription<geometry_msgs::msg::Wrench>(
        "/ft_sensor", 10, std::bind(&PerceptionManager::ft_sensor_callback, this, std::placeholders::_1));

    // F/T 센서 데이터 서비스 생성
    get_ft_data_service_ = this->create_service<perception_interfaces::srv::GetFtData>(
        "perception_manager/get_ft_data",
        std::bind(&PerceptionManager::get_ft_data_callback, this, std::placeholders::_1, std::placeholders::_2));
    
    // 주기적으로 TF를 조회하고 업데이트하는 타이머
    auto timer_period = 100ms;
    object_info_timer_ = this->create_wall_timer(
        timer_period, std::bind(&PerceptionManager::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "Perception Manager Node has been started.");
}

void PerceptionManager::timer_callback()
{
    // 등록된 모든 오브젝트에 대해 TF를 조회
    for (const auto& pair : object_id_to_name_map_)
    {
        int tag_id = pair.first;
        std::string object_name = pair.second;

        try
        {
            // base_link 기준으로 오브젝트의 트랜스폼을 찾음
            geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                "base_link", object_name, tf2::TimePointZero);

            // 오브젝트 데이터 업데이트
            object_data_map_[tag_id].current_transform = transform_stamped;
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Could not transform from 'base_link' to '%s': %s", object_name.c_str(), ex.what());
        }
    }
}

void PerceptionManager::get_object_info_callback(
    const std::shared_ptr<perception_interfaces::srv::GetObjectInfo::Request> request,
    const std::shared_ptr<perception_interfaces::srv::GetObjectInfo::Response> response)
{
    int requested_id = request->object_id;

    // === 주요 변경점: object_data_map_에서 데이터 확인 ===
    if (object_data_map_.count(requested_id) == 0)
    {
        response->success = false;
        RCLCPP_WARN(this->get_logger(), "Object with ID %d not configured.", requested_id);
        return;
    }

    const auto& object_data = object_data_map_.at(requested_id);
    std::string target_frame = object_data.frame_id;
    std::string source_frame = "world";

    response->success = true;
    response->object_pose.header.stamp = this->now();
    response->object_pose.header.frame_id = object_data.frame_id;
    response->object_pose.pose.position.x = object_data.current_transform.transform.translation.x;
    response->object_pose.pose.position.y = object_data.current_transform.transform.translation.y;
    response->object_pose.pose.position.z = object_data.current_transform.transform.translation.z;
    response->object_pose.pose.orientation = object_data.current_transform.transform.rotation;

    response->grasping_offset = object_data.grasping_offset;

    RCLCPP_INFO(this->get_logger(), "Object ID %d: Name='%s', Frame ID='%s'",
                requested_id, object_data.name.c_str(), object_data.frame_id.c_str());
    RCLCPP_INFO(this->get_logger(), "Object position for ID %d: x=%.2f, y=%.2f, z=%.2f",
                requested_id,
                response->object_pose.pose.position.x,
                response->object_pose.pose.position.y,
                response->object_pose.pose.position.z);
    RCLCPP_INFO(this->get_logger(), "Object orientation for ID %d: x=%.2f, y=%.2f, z=%.2f, w=%.2f",
                requested_id,
                response->object_pose.pose.orientation.x,
                response->object_pose.pose.orientation.y,
                response->object_pose.pose.orientation.z,
                response->object_pose.pose.orientation.w);

    RCLCPP_INFO(this->get_logger(), "Grasping orientation for object ID %d: x=%.2f, y=%.2f, z=%.2f, w=%.2f",
                requested_id,
                response->grasping_offset.orientation.x,
                response->grasping_offset.orientation.y,
                response->grasping_offset.orientation.z,
                response->grasping_offset.orientation.w);

    RCLCPP_INFO(this->get_logger(), "Grasping offset for object ID %d: x=%.2f, y=%.2f, z=%.2f",
                requested_id,
                response->grasping_offset.position.x,
                response->grasping_offset.position.y,
                response->grasping_offset.position.z);
    

}

// F/T 센서 콜백 함수 구현
void PerceptionManager::ft_sensor_callback(const geometry_msgs::msg::Wrench::SharedPtr msg)
{
    latest_ft_data_ = *msg;
}

// F/T 센서 데이터 서비스 콜백 함수 구현
void PerceptionManager::get_ft_data_callback(
    const std::shared_ptr<perception_interfaces::srv::GetFtData::Request> request,
    const std::shared_ptr<perception_interfaces::srv::GetFtData::Response> response)
{
    (void)request;
    response->ft_data = latest_ft_data_;
    response->success = true;
    RCLCPP_INFO(this->get_logger(), "Provided latest F/T sensor data.");
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<PerceptionManager>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}