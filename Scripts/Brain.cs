using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Replay
{
    public List<double> states;
    public double reward;

    public Replay(double xr, double ballz, double ballvx, double r)
    {
        states = new List<double>();
        states.Add(xr);
        states.Add(ballz);
        states.Add(ballvx);
        reward = r;
    }
}

public class Brain : MonoBehaviour
{
    public GameObject ball;

    ANN ann;

    float reward = 0.0f;
    List<Replay> replayMemory = new List<Replay>();
    int mCapacity = 10000;

    float discount = 0.99f;
    float exploreRate = 100.0f;
    float maxExploreRate = 100.0f;
    float minExploreRate  = 0.01f;
    float exploreDecay = 0.0001f;

    Vector3 ballStartPos;
    int failCount = 0;
    float tiltSpeed = 0.5f;

    float timer = 0;
    float maxBalanceTime = 0;

    void Start()
    {
        ann = new ANN(3, 2, 1, 6, 0.2f);
        ballStartPos = ball.transform.position;
        Time.timeScale = 5.0f;
    }

    GUIStyle guiStyle = new();
    void OnGUI()
    {
        guiStyle.fontSize = 25;
        guiStyle.normal.textColor = Color.white;
        GUI.BeginGroup(new Rect(10, 10, 600, 150));
        GUI.Box(new Rect(0, 0, 140, 140), "Stats", guiStyle);
        GUI.Label(new Rect(10, 25, 500, 30), "Fails: " + failCount, guiStyle);
        GUI.Label(new Rect(10, 50, 500, 30), "Decay Rate: " + exploreRate, guiStyle);
        GUI.Label(new Rect(10, 75, 500, 30), "Last Best Balance: " + maxBalanceTime, guiStyle);
        GUI.Label(new Rect(10, 100, 500, 30), "This Balance: " + timer, guiStyle);
        GUI.EndGroup();
    }

    void Update()
    {
        if(Input.GetKeyDown("space"))
        {
            ResetBall();
        }
    }

    void FixedUpdate()
    {
        timer += Time.deltaTime;
        List<double> states = new();
        List<double> qs = new();

        states.Add(transform.rotation.x);
        states.Add(transform.position.z);
        states.Add(ball.GetComponent<Rigidbody>().angularVelocity.x);

        qs = SoftMax(ann.CalcOutput(states));
        double maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);
        exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);

        if(maxQIndex == 0)
        {
            transform.Rotate(Vector3.right, tiltSpeed * (float)qs[maxQIndex]);
        }
        else if(maxQIndex == 1)
        {
            transform.Rotate(Vector3.right, -tiltSpeed * (float)qs[maxQIndex]);
        }

        if(ball.GetComponent<BallState>().dropped)
        {
            reward = -1.0f;
        }
        else{
            reward = 0.1f;
        }

        Replay lastMemory = new Replay(transform.rotation.x, ball.transform.position.z, ball.GetComponent<Rigidbody>().angularVelocity.x, reward);

        if(replayMemory.Count > mCapacity)
        {
            replayMemory.RemoveAt(0);
        }
        replayMemory.Add(lastMemory);

        if(ball.GetComponent<BallState>().dropped)
        {
            for(int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<double> toutputsOld = new();
                List<double> toutputsNew = new();
                toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));

                double maxQOld = toutputsOld.Max();
                int action = toutputsOld.ToList().IndexOf(maxQOld);

                double feedback;
                if(i == replayMemory.Count - 1 || replayMemory[i].reward == -1)
                {
                    feedback = replayMemory[i].reward;
                }
                else
                {
                    toutputsNew = SoftMax(ann.CalcOutput(replayMemory[i+1].states));
                    maxQ = toutputsNew.Max();
                    feedback = replayMemory[i].reward + discount * maxQ;
                }

                toutputsOld[action] = feedback;
                ann.Train(replayMemory[i].states, toutputsOld);
            }

            if(timer > maxBalanceTime){maxBalanceTime = timer;}

            timer = 0;

            ball.GetComponent<BallState>().dropped = false;
            transform.rotation = Quaternion.identity;
            ResetBall();
            replayMemory.Clear();
            failCount++;
        }
    }

    void ResetBall()
    {
        transform.position = ballStartPos;
        ball.GetComponent<Rigidbody>().linearVelocity = new Vector3(0, 0, 0);
        ball.GetComponent<Rigidbody>().angularVelocity = new Vector3(0, 0, 0);
    }

    List<double> SoftMax(List<double> values)
    {
        double max = values.Max();

        float scale = 0.0f;

        for(int i = 0; i < values.Count; i++)
        {
            scale += Mathf.Exp((float)(values[i] - max));
        }

        List<double> result = new();
        for(int i = 0; i < values.Count; i++)
        {
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);
        }

        return result;
    }
}
