#include "stringFormatter.h"
#include <chrono>

namespace CTL {

class Watches
{
public:
    Watches()
    {
        using namespace std::chrono;
        timestamp = high_resolution_clock::now();
        // high_resolution
    }

    void reset()
    {
        using namespace std::chrono;
        timestamp = high_resolution_clock::now();
    }

    bool pressed = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> now;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

    std::string textWithTimeFromLastReset(std::string txt = "")
    {
        std::chrono::milliseconds ms = millisecondsFromTimestamp(false);
        float milliSeconds = ms.count() / 1000.0;
        std::string msg = io::xprintf("%s %0.2f", txt.c_str(), milliSeconds);
        return msg;
    }

    std::chrono::milliseconds millisecondsFromTimestamp(bool setNewTimestamp)
    {
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - timestamp);
        if(setNewTimestamp)
        {
            reset();
        }
        return ms;
    }

    void press(std::string txt = "")
    {
        using namespace std::chrono;
        if(!pressed)
        {
            pressed = true;
            LOGI << io::xprintf("%s", txt.c_str());
            lastTime = high_resolution_clock::now();
        } else
        {
            now = high_resolution_clock::now();
            duration<double> xxx = duration_cast<duration<double>>(now - lastTime);
            LOGI << io::xprintf("%s %0.3fs", txt.c_str(), xxx.count());
            lastTime = now;
        }
    };

    void pressE(std::string txt = "")
    {
        using namespace std::chrono;
        if(!pressed)
        {
            pressed = true;
            LOGE << io::xprintf("%s", txt.c_str());
            lastTime = high_resolution_clock::now();
        } else
        {
            now = high_resolution_clock::now();
            duration<double> xxx = duration_cast<duration<double>>(now - lastTime);
            LOGE << io::xprintf("%s %0.3fs", txt.c_str(), xxx.count());
            lastTime = now;
        }
    };
};
} // namespace CTL
