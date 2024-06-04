/****************************************************************************
 *   Copyright (C) 2006-2013 by Jason Ansel, Kapil Arya, and Gene Cooperman *
 *   jansel@csail.mit.edu, kapil@ccs.neu.edu, gene@ccs.neu.edu              *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

/****************************************************************************
 * Coordinator code logic:                                                  *
 * main calls eventLoop, a top-level event loop.                            *
 * eventLoop calls:  onConnect, onData, onDisconnect, startCheckpoint       *
 *   when client or dmtcp_command talks to coordinator.                     *
 * onConnect called on msg at listener port.  It passes control to:         *
 *   handleUserCommand, which takes single char arg ('s', 'c', 'k', 'q', ...)*
 * handleUserCommand calls broadcastMessage to send data back               *
 * any message sent by broadcastMessage takes effect only on returning      *
 *   back up to top level monitorSockets                                    *
 * Hence, even for checkpoint, handleUserCommand just changes state,        *
 *   broadcasts an initial checkpoint command, and then returns to top      *
 *   level.  Replies from clients then drive further state changes.         *
 * The prefix command 'b' (blocking) from dmtcp_command modifies behavior   *
 *   of 'c' so that the reply to dmtcp_command happens only when clients    *
 *   are back in RUNNING state.                                             *
 * onData called when a message arrives at a client's port.  It either      *
 *   processes a per-client special request, or continues the protocol      *
 *   for a checkpoint or restart sequence (see below).                      *
 *                                                                          *
 * updateMinimumState() is responsible for keeping track of states.         *
 * The coordinator keeps a ComputationStatus, with minimumState and         *
 *   maximumState for states of all workers, accessed through getStatus()   *
 *   or through minimumState()                                              *
 * The states for a worker (client) are:                                    *
 * Checkpoint: RUNNING -> SUSPENDED -> CHECKPOINTING                        *
 *                     -> (Checkpoint barriers) -> CHECKPOINTED             *
 *                     -> (Resume barriers) -> RUNNING                      *
 *             [State returns to UNKNOWN if no active workers.]
 *                                                                          *
 * Restart:    RESTARTING -> (Restart barriers) -> RUNNING                  *
 * If debugging, set gdb breakpoint on:                                     *
 *   DmtcpCoordinator::onConnect                                            *
 *   DmtcpCoordinator::onData                                               *
 *   DmtcpCoordinator::handleUserCommand                                    *
 *   DmtcpCoordinator::broadcastMessage                                     *
 ****************************************************************************/

#include "trecr_coordinator.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <iomanip>
#include "../jalib/jassert.h"
#include "../jalib/jconvert.h"
#include "../jalib/jfilesystem.h"
#include "../jalib/jtimer.h"
#include "constants.h"
#include "dmtcpmessagetypes.h"
#include "lookup_service.h"
#include "protectedfds.h"
#include "restartscript.h"
#include "tokenize.h"
#include "syscallwrappers.h"
#include "util.h"
#undef min
#undef max

#define BINARY_NAME "dmtcp_coordinator"
// support timing by huiru.deng
#define MAX_API_COUNT 10000
#define NAME_LEN 100
#define THRESHOLD 0.2

using namespace dmtcp;

static const char *theHelpMessage =
  "COMMANDS:\n"
  "  l : List connected nodes\n"
  "  s : Print status message\n"
  "  c : Checkpoint all nodes\n"
  "  i : Print current checkpoint interval\n"
  "      (To change checkpoint interval, use dmtcp_command)\n"
  "  k : Kill all nodes\n"
  "  q : Kill all nodes and quit\n"
  "  ? : Show this message\n"
  "\n";

static const char *theUsage =
  "Usage: dmtcp_coordinator [OPTIONS] [port]\n"
  "Coordinates checkpoints between multiple processes.\n\n"
  "Options:\n"
  "  -p, --coord-port PORT_NUM (environment variable DMTCP_COORD_PORT)\n"
  "      Port to listen on (default: " STRINGIFY(DEFAULT_PORT) ")\n"
  "  --port-file filename\n"
  "      File to write listener port number.\n"
  "      (Useful with '--port 0', which is used to assign a random port)\n"
  "  --ckptdir (environment variable DMTCP_CHECKPOINT_DIR):\n"
  "      Directory to store dmtcp_restart_script.sh (default: ./)\n"
  "  --tmpdir (environment variable DMTCP_TMPDIR):\n"
  "      Directory to store temporary files (default: env var TMDPIR or /tmp)\n"
  "  --exit-on-last\n"
  "      Exit automatically when last client disconnects\n"
  "  --exit-after-ckpt\n"
  "      Kill peer processes of computation after first checkpoint is created\n"
  "  --daemon\n"
  "      Run silently in the background after detaching from the parent "
  "process.\n"
  "  -i, --interval (environment variable DMTCP_CHECKPOINT_INTERVAL):\n"
  "      Time in seconds between automatic checkpoints\n"
  "      (default: 0, disabled)\n"
  "  --coord-logfile PATH (environment variable DMTCP_COORD_LOG_FILENAME\n"
  "              Coordinator will dump its logs to the given file\n"
  "  -q, --quiet \n"
  "      Skip startup msg; Skip NOTE msgs; if given twice, also skip WARNINGs\n"
  "  --help:\n"
  "      Print this message and exit.\n"
  "  --version:\n"
  "      Print version information and exit.\n"
  "\n"
  "COMMANDS:\n"
  "      type '?<return>' at runtime for list\n"
  "\n"
  HELP_AND_CONTACT_INFO
  "\n";


static int thePort = -1;
static string thePortFile;

static bool exitOnLast = false;
static bool blockUntilDone = false;
static bool exitAfterCkpt = false;
static bool exitAfterCkptOnce = false;
static int blockUntilDoneRemote = -1;
static int checkpoint_count = 0;
static bool mpiMode = false;

static DmtcpCoordinator prog;

/* The coordinator can receive a second checkpoint request while processing the
 * first one.  If the second request comes at a point where the coordinator has
 * broadcast DMT_DO_SUSPEND message but the workers haven't replied, the
 * coordinator sends another DMT_DO_SUSPEND message.  The workers, having
 * replied to the first DMTCP_DO_SUSPEND message (by suspending all the user
 * threads), are waiting for the next message (DMT_DO_FD_LEADER_ELECTION or
 * DMT_KILL_PEER), however they receive DMT_DO_SUSPEND message and thus exit()
 * indicating an error.
 * The fix to this problem is to introduce a global
 * variable "workersRunningAndSuspendMsgSent" which, as the name implies,
 * indicates that the DMT_DO_SUSPEND message has been sent and the coordinator
 * is waiting for replies from the workers.  If this variable is set, the
 * coordinator will not process another checkpoint request.
*/
static bool workersRunningAndSuspendMsgSent = false;

static bool killInProgress = false;
static bool uniqueCkptFilenames = false;

// support timing by huiru.deng
char* ori_log_path = "/Absolute_path_of_CRAC/timing/test.txt"; // original log file
char* dst_log_path = "/Absolute_path_of_CRAC/timing/profiled_log.txt"; // profiled log file
char* dst_log_path_1 = "/Absolute_path_of_CRAC/timing/count_log.txt"; // each api execution count
char* dst_log_path_2 = "/Absolute_path_of_CRAC/timing/last_api.txt"; // last api name of current application

/* If dmtcp_launch/dmtcp_restart specifies '-i', theCheckpointInterval
 * will be reset accordingly (valid for current computation).  If dmtcp_command
 * specifies '-i' (or if user interactively invokes 'i' in coordinator),
 * then both theCheckpointInterval and theDefaultCheckpointInterval are set.
 * A value of '0' means:  never checkpoint (manual checkpoint only).
 */
static uint32_t theCheckpointInterval = 0; // Current checkpoint interval
static uint32_t theDefaultCheckpointInterval = 0; // Reset to this on new comp.
static bool timerExpired = false;

static void resetCkptTimer();

const int STDIN_FD = fileno(stdin);

JTIMER(checkpoint);
JTIMER(restart);

static int workersAtCurrentBarrier = 0;
static string currentBarrier;
static string prevBarrier;
static ssize_t eventId = 0;

static UniquePid compId;
static int numPeers = -1;
static time_t curTimeStamp = -1;
static time_t ckptTimeStamp = -1;

static LookupService lookupService;
static bool writeKvData = false;

static string coordHostname;
static struct in_addr localhostIPAddr;

static char *tmpDir = NULL;
static string ckptDir;

#define MAX_EVENTS 10000
struct epoll_event events[MAX_EVENTS];
int epollFd;
static jalib::JSocket *listenSock = NULL;

static void removeStaleSharedAreaFile();
static void preExitCleanup();

static pid_t _nextVirtualPid = INITIAL_VIRTUAL_PID;

static int theNextClientNumber = 1;
vector<CoordClient *>clients;

// support timing by huiru.deng
/*-------------------------------------Start Log Profile function by huiru.deng----------------------------------*/
std::map<int, std::tuple<string, float, int>> ori_api_list;
// unordered_map<string, int> map_api;
std::map<string, int> map_count;
int total_index;
bool api_visit[MAX_API_COUNT];
std::map<float, bool> interval_visit;
std::map<string, int> api_total_count;
int order;
int result_order;

void LogProfile::init() 
{
  total_index = 0;
  order = 0;
  result_order = 0;
  ori_api_list.clear();
  map_count.clear();
  api_total_count.clear();
  memset(api_visit, sizeof(bool)*MAX_API_COUNT, false);
  interval_visit.clear();
}
/*
int get(string name) {
	auto iter = map_api.find(name);
	if (iter == map_api.end()) {
		map_api[name] = total_index;
		return total_index++;
	}
	return map_api[name];
}*/

float LogProfile::pre_process(char* str)
{
  int len = strlen(str);
  float value = 0;
  if (str[len - 2] == 'm' && str[len - 1] == 's')
  {
    int i = 0;
    while (str[i] != '.') {
      value = value * 10 + (str[i] - '0');
      i++;
    }
    return value / 1000;
  }
  else 
  {
    char tem_str[NAME_LEN];
    int i = 0;
    while (str[i] != 's')
    {
      tem_str[i] = str[i];
      i++;
    }
    return atof(tem_str);
  }
}

void LogProfile::generate_ori_data(string api_name, float time) 
{
  map_count[api_name]++;
  auto tup1 = std::make_tuple(api_name, time, map_count[api_name]);
  ori_api_list.insert({ order, tup1 });
  order++;
}

float LogProfile::get_interval(std::map<int, std::tuple<string, float, int>>::iterator iter, float interval, float init_interval)
{
  float value = interval;
  if (static_cast<int>(std::get<1>(iter->second) / interval) == 0)
  {
    value = (static_cast<int>(std::get<1>(iter->second) / init_interval) + 1)*init_interval;
    return value;
  }
  else
  {
    value = (static_cast<int>(std::get<1>(iter->second) / init_interval))*init_interval;
    return value;
  }
}

void LogProfile::profile_log(float init_interval, char* ori_log_path, char* dst_log_path)
{
  init();
  // parse ori api list
  FILE* fp = fopen(ori_log_path, "r");
  FILE* fp_out = fopen(dst_log_path, "w");
  FILE* fp_out_1 = fopen(dst_log_path_1, "w");
  FILE* fp_out_2 = fopen(dst_log_path_2, "w");
  char api_name[NAME_LEN];
  char execution_time_str[NAME_LEN];
  int execute_count;

  while (fscanf(fp, "%s %s", &execution_time_str, &api_name) != EOF)
  {
    float real_time = pre_process(execution_time_str);
    string tem_api_name = api_name;
    api_total_count[tem_api_name]++;
    generate_ori_data(tem_api_name, real_time);
  }

  // traverse the map to get the result
  std::map<int, std::tuple<string, float, int>>::iterator iter;
  std::map<int, std::tuple<string, float, int>>::reverse_iterator iter_tem;
  iter = ori_api_list.begin();
  iter_tem = ori_api_list.rbegin();
  float interval = init_interval;

  // save last name
  fprintf(fp_out_2, "%s\n", std::get<0>(iter_tem->second).c_str());

  while (iter != ori_api_list.end())
  {
    // 1. get current interval range
    interval = get_interval(iter, interval, init_interval);

    // 2. record checkpoint info
    // situation 1 : no high workload, interval unreached
    if (std::get<1>(iter->second) < interval && std::get<0>(iter->second) != "cuMemcpyHtoDAsync")
    {
      iter++;
      continue;
    }
    // situation 2 : interval unreached, high workload meets
    else if (std::get<1>(iter->second) < interval && std::get<0>(iter->second) == "cuMemcpyHtoDAsync")
    {
      float threshold = (interval - std::get<1>(iter->second)) / interval;
      if (threshold < THRESHOLD)
      {
        // do checkpoint before cuMemcpyHtoD
        int pre_api_index = (iter->first) - 1;
        while (std::get<0>(ori_api_list[pre_api_index]) == "cuMemcpyHtoDAsync")
        {
          pre_api_index = pre_api_index - 1;
        }
        if (api_visit[pre_api_index] == true)
        {
          iter++;
          continue;
        }
        else
        {
          api_visit[pre_api_index] = true;
          std::tuple<string, float, int> pre_api_info = ori_api_list[pre_api_index];
          fprintf(fp_out, "%d %s %d\n", result_order++, std::get<0>(pre_api_info).c_str(), std::get<2>(pre_api_info));
          iter++;
          continue;
        }
      }
      else
      {
        iter++;
        continue;
      }
    }
      // situation 3 : interval reached
    else if (std::get<1>(iter->second) >= interval)
    {
      if (interval_visit[interval] == true)
      {
        iter++;
        continue;
      }

      bool flag = false;
      int pre_api_index = iter->first - 1;

      while ( ((std::get<1>(iter->second) - std::get<1>(ori_api_list[pre_api_index]))/std::get<1>(iter->second)) <= THRESHOLD )
      {
        if (std::get<0>(ori_api_list[pre_api_index]) == "cuMemcpyHtoDAsync")
        {
          flag = true;
          break;
        }
        pre_api_index = pre_api_index - 1;
      }

      if (flag == true)
      {
        iter++;
        continue;
      }
      else
      {
        interval_visit[interval] = true;
        fprintf(fp_out, "%d %s %d\n", result_order++, std::get<0>(ori_api_list[iter->first - 1]).c_str(), std::get<2>(ori_api_list[iter->first - 1]));
        iter++;
        continue;
      }
    }
  }

  // save the api_total_count to a file
  std::map<string, int>::iterator iter1;
  iter1 = api_total_count.begin();
  while (iter1 != api_total_count.end())
  {
    fprintf(fp_out_1, "%s %d\n", iter1->first.c_str(), iter1->second);
    iter1++;
  }

  fclose(fp);
  fclose(fp_out);
  fclose(fp_out_1);
  fclose(fp_out_2);
}
/*--------------------------------------End of profile log code by huiru.deng-------------------------------------*/

CoordClient::CoordClient(const jalib::JSocket &sock,
                         const struct sockaddr_storage *addr,
                         socklen_t len,
                         DmtcpMessage &hello_remote,
                         int isNSWorker)
  : _sock(sock),
    _barrier("")
{
  _isNSWorker = isNSWorker;
  _realPid = hello_remote.realPid;
  _clientNumber = theNextClientNumber++;
  _identity = hello_remote.from;
  _state = hello_remote.state;
  struct sockaddr_in *in = (struct sockaddr_in *)addr;
  _ip = inet_ntoa(in->sin_addr);
}

void
CoordClient::readProcessInfo(DmtcpMessage &msg)
{
  if (msg.extraBytes > 0) {
    char *extraData = new char[msg.extraBytes];
    _sock.readAll(extraData, msg.extraBytes);
    _hostname = extraData;
    _progname = extraData + _hostname.length() + 1;
    delete[] extraData;
  }
}

pid_t
DmtcpCoordinator::getNewVirtualPid()
{
  pid_t pid = -1;

  JASSERT(_virtualPidToClientMap.size() < MAX_VIRTUAL_PID / 1000)
  .Text("Exceeded maximum number of processes allowed");
  while (1) {
    pid = _nextVirtualPid;
    _nextVirtualPid += 1000;
    if (_nextVirtualPid > MAX_VIRTUAL_PID) {
      _nextVirtualPid = INITIAL_VIRTUAL_PID;
    }
    if (_virtualPidToClientMap.find(pid) == _virtualPidToClientMap.end()) {
      break;
    }
  }
  JASSERT(pid != -1).Text("Not Reachable");
  return pid;
}

static string replyData = "";

void
DmtcpCoordinator::handleUserCommand(char cmd, DmtcpMessage *reply /*= NULL*/)
{
  if (reply != NULL) {
    reply->coordCmdStatus = CoordCmdStatus::NOERROR;
  }

  switch (cmd) {
  case 'b': case 'B':  // prefix blocking command, prior to checkpoint command
    JTRACE("blocking checkpoint beginning...");
    blockUntilDone = true;
    break;
  case 'x': case 'X':  // prefix exit command, prior to checkpoint command
    JTRACE("Will exit after creating the checkpoint...");
    exitAfterCkptOnce = true;
    break;
  case 'c': case 'C':
    JTRACE("checkpointing...");
    if (startCheckpoint()) {
      if (reply != NULL) {
        reply->numPeers = getStatus().numPeers;
      }
    } else {
      if (reply != NULL) {
        reply->coordCmdStatus = CoordCmdStatus::ERROR_NOT_RUNNING_STATE;
      }
    }
    break;
  case 'i': case 'I':
    JTRACE("setting checkpoint interval...");
    updateCheckpointInterval(theCheckpointInterval);
    if (theCheckpointInterval == 0) {
      printf("Current Checkpoint Interval:"
             " Disabled (checkpoint manually instead)\n");
    } else {
      printf("Current Checkpoint Interval: %d\n", theCheckpointInterval);
    }
    if (theDefaultCheckpointInterval == 0) {
      printf("Default Checkpoint Interval:"
             " Disabled (checkpoint manually instead)\n");
    } else {
      printf("Default Checkpoint Interval: %d\n", theDefaultCheckpointInterval);
    }
    break;
  case 'l': case 'L':
  case 't': case 'T':
    if (reply != NULL) {
      replyData = printList();
      reply->extraBytes = replyData.length();
    } else {
      JASSERT_STDERR << printList();
    }
    break;
  case 'u': case 'U':
  {
    JASSERT_STDERR << "Host List:\n";
    JASSERT_STDERR << "HOST => # connected clients \n";
    dmtcp::map<string, int>clientHosts;
    for (size_t i = 0; i < clients.size(); i++) {
      if (clientHosts.find(clients[i]->hostname()) == clientHosts.end()) {
        clientHosts[clients[i]->hostname()] = 1;
      } else {
        clientHosts[clients[i]->hostname()] += 1;
      }
    }
    for (dmtcp::map<string, int>::iterator it = clientHosts.begin();
         it != clientHosts.end();
         ++it) {
      JASSERT_STDERR << it->first << " => " << it->second << '\n';
    }
    break;
  }
  case 'q': case 'Q':
  {
    JNOTE("killing all connected peers and quitting ...");
    broadcastMessage(DMT_KILL_PEER);
    JASSERT_STDERR << "DMTCP coordinator exiting... (per request)\n";
    for (size_t i = 0; i < clients.size(); i++) {
      clients[i]->sock().close();
    }
    listenSock->close();
    preExitCleanup();
    JTRACE("Exiting ...");
    recordEvent("Exiting");
    serializeKVDB();
    exit(0);
    break;
  }
  case 'k': case 'K':
    JNOTE("Killing all connected Peers...");

    // FIXME: What happens if a 'k' command is followed by a 'c' command before
    // the *real* broadcast takes place?         --Kapil
    broadcastMessage(DMT_KILL_PEER);
    break;
  case 'h': case 'H': case '?':
    JASSERT_STDERR << theHelpMessage;
    break;
  case 's': case 'S':
  {
    ComputationStatus s = getStatus();
    bool running = (s.minimumStateUnanimous &&
                    s.minimumState == WorkerState::RUNNING);
    if (reply != NULL) {
      reply->numPeers = s.numPeers;
      reply->isRunning = running;
      reply->theCheckpointInterval = theCheckpointInterval;
    } else {
      printStatus(s.numPeers, running);
    }
    break;
  }
  case ' ': case '\t': case '\n': case '\r':

    // ignore whitespace
    break;
  default:
    JTRACE("unhandled user command")(cmd);
    if (reply != NULL) {
      reply->coordCmdStatus = CoordCmdStatus::ERROR_INVALID_COMMAND;
    }
  }
}

void
DmtcpCoordinator::writeSubmissionHostInfo() {
  fprintf(stderr, "Write host info\n");
  ofstream o;
  string home = getenv("HOME");
  home = home + "/.coordinfo";
  o.open(home.c_str(), std::ios::out | std::ios::trunc);
  if (o.fail()) {
    fprintf(stderr, "%s", "Failed to truncate and open ~/.coordinfo\n");
    exit(1);
  }
  o << "Status..." << std::endl
    << "Host: " << coordHostname
    << " (" << inet_ntoa(localhostIPAddr) << ")" << std::endl
    << "Port: " << thePort << std::endl
    << "PID: " << getpid() << std::endl
    << "Checkpoint Interval: ";
  if (theCheckpointInterval == 0) {
    o << "disabled (checkpoint manually instead)" << std::endl;
  } else {
    o << theCheckpointInterval << std::endl;
  }
  o.close();
}

void
DmtcpCoordinator::printStatus(size_t numPeers, bool isRunning)
{
  ostringstream o;

  o << "Status..." << std::endl
    << "Host: " << coordHostname
    << " (" << inet_ntoa(localhostIPAddr) << ")" << std::endl
    << "Port: " << thePort << std::endl
    << "Checkpoint Interval: ";

  if (theCheckpointInterval == 0) {
    o << "disabled (checkpoint manually instead)" << std::endl;
  } else {
    o << theCheckpointInterval << std::endl;
  }

  o << "Exit on last client: " << exitOnLast << std::endl
    << "Exit after checkpoint: " << exitAfterCkpt << std::endl

    // << "Exit after checkpoint (first time only): " << exitAfterCkptOnce
    // << std::endl
    << "Computation Id: " << compId << std::endl
    << "Checkpoint Dir: " << ckptDir << std::endl
    << "NUM_PEERS=" << numPeers << std::endl
    << "RUNNING=" << (isRunning ? "yes" : "no") << std::endl;
  printf("%s", o.str().c_str());
  fflush(stdout);
}

string
DmtcpCoordinator::printList()
{
  ostringstream o;

  o << "Client List:\n";
  o << "#, PROG[virtPID:realPID]@HOST, DMTCP-UNIQUEPID, STATE, BARRIER\n";
  for (size_t i = 0; i < clients.size(); i++) {
    o << clients[i]->clientNumber()
      << ", " << clients[i]->progname()
      << "[" << clients[i]->identity().pid() << ":" << clients[i]->realPid()
      << "]@" << clients[i]->hostname()
#ifdef PRINT_REMOTE_IP
      << "(" << clients[i]->ip() << ")"
#endif // ifdef PRINT_REMOTE_IP
      << ", " << clients[i]->identity()
      << ", " << clients[i]->state()
      << ", " << clients[i]->barrier()
      << '\n';
  }
  return o.str();
}

void
DmtcpCoordinator::recordEvent(string const& event)
{
  eventId++;
  ostringstream o;
  o << std::setfill('0') << std::setw(5) << eventId << "-" << event;
  lookupService.set("/Event_Timestamp_Ms", Util::getTimestampStr(), o.str());
}

void
DmtcpCoordinator::serializeKVDB()
{
}

void
DmtcpCoordinator::releaseBarrier(const string &barrier)
{
  ComputationStatus status = getStatus();

  JTRACE("total Peers:")(numPeers)(barrier.c_str());
  // mod by tian01.liu, force to wait until the restored process number equeals the numPeers-- for temp
  if (strstr(barrier.c_str(), "RESTART") && status.numPeers != numPeers) {
    return;
  }

  if (workersAtCurrentBarrier == status.numPeers /*&& status.numPeers == 4*/) {
    JTRACE("Releasing barrier") (barrier);
    prevBarrier = currentBarrier;
    currentBarrier.clear();
    workersAtCurrentBarrier = 0;

    _numCkptWorkers = status.numPeers;
    broadcastMessage(DMT_BARRIER_RELEASED,
                     prevBarrier.length() + 1,
                     prevBarrier.c_str());
    if (status.minimumState == WorkerState::RUNNING) {
      JNOTE("Checkpoint complete; all workers running");
      resetCkptTimer();
    }
  }
}

void
DmtcpCoordinator::processBarrier(const string &barrier)
{
  // Check if this is the first process to reach barrier.
  if (currentBarrier.empty()) {
    currentBarrier = barrier;
  } else {
    JASSERT(barrier == currentBarrier) (barrier) (currentBarrier);
  }

  ++workersAtCurrentBarrier;

  releaseBarrier(barrier);
}


void
DmtcpCoordinator::recordCkptFilename(CoordClient *client, const char *extraData)
{
  client->setState(WorkerState::CHECKPOINTED);
  JASSERT(extraData != NULL)
  .Text("extra data expected with DMT_CKPT_FILENAME message");

  string ckptFilename = extraData;
  string hostname = extraData + ckptFilename.length() + 1;
  string shellType;

  ckptFilename = extraData;
  shellType = extraData + ckptFilename.length() + 1;
  hostname = extraData + shellType.length() + 1 + ckptFilename.length() + 1;

  JTRACE("recording restart info") (ckptFilename) (hostname);
  JTRACE("recording restart info with shellType")
    ( ckptFilename ) ( hostname ) ( shellType );
  if (shellType.empty())
    _restartFilenames[hostname].push_back(ckptFilename);
  else if (shellType == "rsh")
    _rshCmdFileNames[hostname].push_back(ckptFilename);
  else if(shellType == "ssh")
    _sshCmdFileNames[hostname].push_back(ckptFilename);
  else 
  {
    JASSERT(0)(shellType)
      .Text("Shell command not supported. Report this to DMTCP community.");
  }
  _numRestartFilenames++;

  if (_numRestartFilenames == _numCkptWorkers) {
    const string restartScriptPath =
      RestartScript::writeScript(ckptDir,
                                 uniqueCkptFilenames,
                                 ckptTimeStamp,
                                 theCheckpointInterval,
                                 thePort,
                                 compId,
                                 _restartFilenames,
                                 _rshCmdFileNames,
                                 _sshCmdFileNames);

    JTRACE("Checkpoint complete. Wrote restart script") (restartScriptPath);

    JTIMER_STOP(checkpoint);
    recordEvent("Ckpt-Complete");
    serializeKVDB();

    if (blockUntilDone) {
      DmtcpMessage blockUntilDoneReply(DMT_USER_CMD_RESULT);
      JNOTE("replying to dmtcp_command:  we're done");

      // These were set in DmtcpCoordinator::onConnect in this file
      jalib::JSocket remote(blockUntilDoneRemote);
      remote << blockUntilDoneReply;
      remote.close();
      blockUntilDone = false;
      blockUntilDoneRemote = -1;
    }

    if (exitAfterCkpt || exitAfterCkptOnce) {
      JTRACE("Checkpoint Done. Killing all peers.");
      broadcastMessage(DMT_KILL_PEER);
      exitAfterCkptOnce = false;
    } else {
      // On checkpoint/resume, we should not be resetting the lookup service.
      //   This is absolutely required by the InfiniBand plugin.
      // lookupService.reset();
    }
    _numRestartFilenames = 0;
    _numCkptWorkers = 0;

    // All the workers have checkpointed so now it is safe to reset this flag.
    workersRunningAndSuspendMsgSent = false;
  }
}

void
DmtcpCoordinator::onData(CoordClient *client)
{
  DmtcpMessage msg;
  const char* interval;
  JASSERT(client != NULL);

  client->sock() >> msg;
  msg.assertValid();
  char *extraData = 0;
  if (msg.extraBytes > 0) {
    extraData = new char[msg.extraBytes];
    client->sock().readAll(extraData, msg.extraBytes);
  }

  WorkerState::eWorkerState prevClientState = client->state();
  client->setState(msg.state);

  switch (msg.type) {
  case DMT_WORKER_RESUMING:
  {
    JTRACE("Worker resuming execution")
      (msg.from) (prevClientState) (msg.state);

    client->setBarrier("");
    // A worker is switching from RESTARTING, stop restart timer.
    // Multiple calls are harmless.
    if (prevClientState == WorkerState::RESTARTING) {
      ComputationStatus s = getStatus();
      if (s.minimumStateUnanimous && s.minimumState == WorkerState::RUNNING) {
        JTIMER_STOP(restart);
        recordEvent("Restart-Complete");
        serializeKVDB();
      }
    }
    break;
  }

  case DMT_BARRIER:
  {
    string barrier = msg.barrier;
    JTRACE("got DMT_BARRIER message")
      (msg.from) (prevClientState) (msg.state) (barrier);

    // Warn if we have two consecutive barriers of the same name.
    JWARNING(barrier != client->barrier()) (barrier) (client->barrier());
    client->setBarrier(barrier);
    processBarrier(barrier);
    break;
  }

  case DMT_UNIQUE_CKPT_FILENAME:
    uniqueCkptFilenames = true;

  // Fall though
  case DMT_CKPT_FILENAME:
    recordCkptFilename(client, extraData);
    break;

  case DMT_GET_CKPT_DIR:
  {
    DmtcpMessage reply(DMT_GET_CKPT_DIR_RESULT);
    reply.extraBytes = ckptDir.length() + 1;
    client->sock() << reply;
    client->sock().writeAll(ckptDir.c_str(), reply.extraBytes);
    break;
  }
  case DMT_UPDATE_CKPT_DIR:
  {
    JASSERT(extraData != 0)
    .Text("extra data expected with DMT_UPDATE_CKPT_DIR message");
    if (strcmp(ckptDir.c_str(), extraData) != 0) {
      ckptDir = extraData;
      JTRACE("Updated ckptDir") (ckptDir);
    }
    break;
  }

  case DMT_UPDATE_PROCESS_INFO_AFTER_FORK:
  {
    JTRACE("Updating process Information after fork()")
      (client->hostname()) (client->progname()) (msg.from) (client->identity());
    client->identity(msg.from);
    client->realPid(msg.realPid);
    break;
  }
  case DMT_UPDATE_PROCESS_INFO_AFTER_INIT_OR_EXEC:
  {
    string progname = extraData;
    JTRACE("Updating process Information after exec()")
      (progname) (msg.from) (client->identity());
    client->setState(msg.state);
    client->progname(progname);
    client->identity(msg.from);
    if (workersRunningAndSuspendMsgSent) {
      // If we received this message from the worker _after_ we broadcasted
      // DMT_DO_CHECKPOINT message to workers, there are two possible scenarios:
      // 1. User thread called exec before ckpt-thread had a chance to read the
      //    DMT_DO_CHECKPOINT message from the coordinator-socket. In this case,
      //    once the exec completes and a new ckpt-thread is created, that
      //    thread will read the pending DMT_DO_CHECKPOINT message and proceed
      //    as expected.
      // 2. The ckpt-thread read the DMT_DO_CHECKPOINT message, but before it
      //    could quiesce user threads, one of them called exec. After
      //    completing exec, the new ckpt-thread won't know that the coordinator
      //    had requested a checkpoint via DMT_DO_CHECKPOINT message. The
      //    ckpt-thread will block until it gets a DMT_DO_CHECKPOINT message
      //    while the coordinator is waiting for this process to respond to the
      //    earlier DMT_DO_CHECKPOINT message, leading to a deadlocked
      //    computation.
      //
      // In order to handle case (2) above, we send a second DMT_DO_CHECKPOINT
      // message to this worker. If this worker already processed the previous
      // DMT_DO_CHECKPOINT message as in case (1) above, it will ignore this
      // duplicate message.

      ResendDoCheckpointMsgToWorker(client);
    }
    break;
  }

  case DMT_KVDB_REQUEST:
  {
    JTRACE("received DMT_KVDB_REQUEST msg") (client->identity());
    lookupService.processRequest(client->sock(), msg, extraData);
    break;
  }

  case DMT_NULL:
    JWARNING(false) (msg.type).Text(
      "unexpected message from worker. Closing connection");
    onDisconnect(client);
    break;
  case DMT_DO_CHECKPOINT2:
    startCheckpoint();
    break;
  case DMT_EXECUTE:
    interval = getenv(ENV_VAR_CKPT_INTR);
    if (interval != NULL) {
      printf("1. start generate initial profile log........................\n");
      LogProfile LogPro;
      LogPro.profile_log(static_cast<float>(jalib::StringToInt(interval)), ori_log_path, dst_log_path);
      printf("1. finish generate initial profile log.......................\n");
      broadcastMessage(DMT_UPDATE_LOG);
    }
    break;
  default:
    JASSERT(false) (msg.from) (msg.type)
    .Text("unexpected message from worker");
  }

  delete[] extraData;
}

static void
removeStaleSharedAreaFile()
{
  ostringstream o;

  o << tmpDir
    << "/dmtcpSharedArea." << compId << "." << std::hex << curTimeStamp;
  JTRACE("Removing sharedArea file.") (o.str());
  unlink(o.str().c_str());
}

static void
preExitCleanup()
{
  removeStaleSharedAreaFile();
  JTRACE("Removing port-file") (thePortFile);
  unlink(thePortFile.c_str());
}

void
DmtcpCoordinator::onDisconnect(CoordClient *client)
{
  if (client->isNSWorker()) {
    client->sock().close();
    delete client;
    return;
  }
  for (size_t i = 0; i < clients.size(); i++) {
    if (clients[i] == client) {
      clients.erase(clients.begin() + i);
      break;
    }
  }
  client->sock().close();
  JTRACE("client disconnected") (client->identity()) (client->progname());
  _virtualPidToClientMap.erase(client->virtualPid());

  ComputationStatus s = getStatus();
  if (s.numPeers < 1) {
    if (exitOnLast) {
      JNOTE("last client exited, shutting down..");
      handleUserCommand('q');
    } else {
      removeStaleSharedAreaFile();
    }

    // If a kill in is progress, the coordinator refuses any new connections,
    // thus we need to reset it to false once all the processes in the
    // computations have disconnected.
    killInProgress = false;
    if (theCheckpointInterval != theDefaultCheckpointInterval) {
      updateCheckpointInterval(theDefaultCheckpointInterval);
      JNOTE("CheckpointInterval reset on end of current computation")
        (theCheckpointInterval);
    }
  } else {
    // If all other workers are at currentBarrier, release it.
    if (!currentBarrier.empty() && client->barrier() == currentBarrier) {
      --workersAtCurrentBarrier;
      releaseBarrier(currentBarrier);
    }
  }

  delete client;
}

void
DmtcpCoordinator::initializeComputation()
{
  JTRACE("Resetting computation");

  // this is the first connection, do some initializations
  workersRunningAndSuspendMsgSent = false;
  killInProgress = false;

  // _nextVirtualPid = INITIAL_VIRTUAL_PID;

  // drop current computation group to 0
  compId = UniquePid(0, 0, 0);
  curTimeStamp = 0; // Drop timestamp to 0
  numPeers = -1; // Drop number of peers to unknown
  blockUntilDone = false;
  exitAfterCkptOnce = false;
  workersAtCurrentBarrier = 0;
  prevBarrier.clear();
  currentBarrier.clear();
}

void
DmtcpCoordinator::onConnect()
{
  struct sockaddr_storage remoteAddr;
  socklen_t remoteLen = sizeof(remoteAddr);
  jalib::JSocket remote = listenSock->accept(&remoteAddr, &remoteLen);

  JTRACE("accepting new connection") (remote.sockfd());

  if (!remote.isValid()) {
    remote.close();
    return;
  }

  DmtcpMessage hello_remote;
  hello_remote.poison();
  JTRACE("Reading from incoming connection...");
  remote >> hello_remote;
  if (!remote.isValid()) {
    remote.close();
    return;
  }

  if (hello_remote.type == DMT_NAME_SERVICE_WORKER) {
    CoordClient *client = new CoordClient(remote, &remoteAddr, remoteLen,
                                          hello_remote);

    addDataSocket(client);
    return;
  }

  if (hello_remote.type == DMT_USER_CMD) {
    // TODO(kapil): Update ckpt interval only if a valid one was supplied to
    // dmtcp_command.
#ifdef TIMING
    printf("3. start update profile log file.......................\n");
    LogProfile LogPro;
    LogPro.profile_log(static_cast<float>(hello_remote.theCheckpointInterval), ori_log_path, dst_log_path);
    printf("3. finish update profile log file.......................\n");

    updateCheckpointInterval(hello_remote.theCheckpointInterval);
    processDmtUserCmd(hello_remote, remote);
    broadcastMessage(DMT_USER_ACT);
#else
    updateCheckpointInterval(hello_remote.theCheckpointInterval);
    processDmtUserCmd(hello_remote, remote);
#endif
    return;
  }

  if (killInProgress) {
    JNOTE("Connection request received in the middle of killing computation. "
          "Sending it the kill message.");
    DmtcpMessage msg;
    msg.type = DMT_KILL_PEER;
    remote << msg;
    remote.close();
    return;
  }

  // If no client is connected to Coordinator, then there can be only zero data
  // sockets OR there can be one data socket and that should be STDIN.
  if (clients.size() == 0) {
    initializeComputation();
  }

  CoordClient *client = new CoordClient(remote, &remoteAddr, remoteLen,
                                        hello_remote);

  if (hello_remote.extraBytes > 0) {
    client->readProcessInfo(hello_remote);
  }

  if (hello_remote.type == DMT_RESTART_WORKER) {
    if (!validateRestartingWorkerProcess(hello_remote, remote,
                                         &remoteAddr, remoteLen)) {
      return;
    }
    client->virtualPid(hello_remote.from.pid());
    _virtualPidToClientMap[client->virtualPid()] = client;
  } else if (hello_remote.type == DMT_NEW_WORKER) {
    // Comping from dmtcp_launch or fork(), ssh(), etc.
    JASSERT(hello_remote.state == WorkerState::RUNNING ||
            hello_remote.state == WorkerState::UNKNOWN);
    JASSERT(hello_remote.virtualPid == -1);
    client->virtualPid(getNewVirtualPid());
    if (!validateNewWorkerProcess(hello_remote, remote, client,
                                  &remoteAddr, remoteLen)) {
      return;
    }
    _virtualPidToClientMap[client->virtualPid()] = client;
  } else {
    JASSERT(false) (hello_remote.type)
    .Text("Connect request from Unknown Remote Process Type");
  }

  updateCheckpointInterval(hello_remote.theCheckpointInterval);
  JTRACE("worker connected") (hello_remote.from) (client->progname());

  clients.push_back(client);
  addDataSocket(client);

  JTRACE("END") (clients.size());
}

void
DmtcpCoordinator::processDmtUserCmd(DmtcpMessage &hello_remote,
                                    jalib::JSocket &remote)
{
  // dmtcp_command doesn't handshake (it is antisocial)
  JTRACE("got user command from dmtcp_command")((char)hello_remote.coordCmd);
  DmtcpMessage reply;
  reply.type = DMT_USER_CMD_RESULT;

  // if previous 'b' blocking prefix command had set blockUntilDone
  if (blockUntilDone && blockUntilDoneRemote == -1 &&
      hello_remote.coordCmd == 'c') {
    // Reply will be done in DmtcpCoordinator::onData in this file.
    blockUntilDoneRemote = remote.sockfd();
    handleUserCommand(hello_remote.coordCmd, &reply);
  } else if (hello_remote.coordCmd == 'i') {
    // theDefaultCheckpointInterval = hello_remote.theCheckpointInterval;
    // theCheckpointInterval = theDefaultCheckpointInterval;
    handleUserCommand(hello_remote.coordCmd, &reply);
    remote << reply;
    remote.close();
  } else {
    handleUserCommand(hello_remote.coordCmd, &reply);
    remote << reply;
    if (reply.extraBytes > 0) {
      remote.writeAll(replyData.c_str(), reply.extraBytes);
    }
    remote.close();
  }
}

/*
 * Returns the current timestamp with nanosecond resolution
 */
static uint64_t
getCurrTimestamp()
{
  struct timespec value;
  uint64_t nsecs = 0;
  JASSERT(clock_gettime(CLOCK_MONOTONIC, &value) == 0);
  nsecs = value.tv_sec*1000000000L + value.tv_nsec;
  return nsecs;
}


bool
DmtcpCoordinator::validateRestartingWorkerProcess(
  DmtcpMessage &hello_remote,
  jalib::JSocket &remote,
  const struct sockaddr_storage *remoteAddr,
  socklen_t remoteLen)
{
  const struct sockaddr_in *sin = (const struct sockaddr_in *)remoteAddr;
  string remoteIP = inet_ntoa(sin->sin_addr);
  DmtcpMessage hello_local(DMT_ACCEPT);

  JASSERT(hello_remote.state == WorkerState::RESTARTING) (hello_remote.state);

  if (compId == UniquePid(0, 0, 0)) {
    lookupService.reset();
    recordEvent("Restarting-Computation");
    JASSERT(minimumState() == WorkerState::UNKNOWN) (minimumState())
    .Text("Coordinator should be idle at this moment");

    // Coordinator is free at this moment - set up all the things
    compId = hello_remote.compGroup;
    numPeers = hello_remote.numPeers;
    curTimeStamp = getCurrTimestamp();
    JTRACE("FIRST dmtcp_restart connection.  Set numPeers. Generate timestamp")
      (numPeers) (curTimeStamp) (compId);
    JTIMER_START(restart);
    recordEvent("Restart-Start");
  } else if (minimumState() != WorkerState::RESTARTING) {
    JTRACE("Computation not in RESTARTING state."
          "  Reject incoming computation process requesting restart.")
      (compId) (hello_remote.compGroup) (minimumState());
    hello_local.type = DMT_REJECT_NOT_RESTARTING;
    remote << hello_local;
    remote.close();
    return false;
  } else if (hello_remote.compGroup != compId) {
    JTRACE("Reject incoming computation process requesting restart,"
          " since it is not from current computation.")
      (compId) (hello_remote.compGroup);
    hello_local.type = DMT_REJECT_WRONG_COMP;
    remote << hello_local;
    remote.close();
    return false;
  }

  // dmtcp_restart already connected and compGroup created.
  // Computation process connection
  JASSERT(curTimeStamp != 0);

  JTRACE("Connection from (restarting) computation process")
    (compId) (hello_remote.compGroup) (minimumState());

  hello_local.coordTimeStamp = curTimeStamp;
  if (Util::strStartsWith(remoteIP.c_str(), "127.")) {
    memcpy(&hello_local.ipAddr, &localhostIPAddr, sizeof localhostIPAddr);
  } else {
    memcpy(&hello_local.ipAddr, &sin->sin_addr, sizeof localhostIPAddr);
  }
  remote << hello_local;

  // NOTE: Sending the same message twice. We want to make sure that the
  // worker process receives/processes the first messages as soon as it
  // connects to the coordinator. The second message will be processed in
  // postRestart routine in DmtcpWorker.
  //
  // The reason to do this is the following. The dmtcp_restart process
  // connects to the coordinator at a very early stage. Later on, before
  // exec()'ing into mtcp_restart, it reconnects to the coordinator using
  // it's original UniquiePid and closes the earlier socket connection.
  // However, the coordinator might process the disconnect() before it
  // processes the connect() which would lead to a situation where the
  // coordinator is not connected to any worker processes. The coordinator
  // would now process the connect() and may reject the worker because the
  // worker state is RESTARTING, but the minimumState() is UNKNOWN.
  // remote << hello_local;

  return true;
}

void
DmtcpCoordinator::ResendDoCheckpointMsgToWorker(CoordClient *client)
{
  JASSERT(workersRunningAndSuspendMsgSent);
  /* Worker trying to connect after SUSPEND message has been sent.
    * This happens if the worker process is executing a fork() or exec() system
    * call when the DMT_DO_SUSPEND is broadcast. We need to make sure that the
    * child process is allowed to participate in the current checkpoint.
    */
  ComputationStatus s = getStatus();
  JASSERT(s.numPeers > 0) (s.numPeers);
  JASSERT(s.minimumState != WorkerState::SUSPENDED) (s.minimumState);

  JNOTE("Sending DMT_DO_CHECKPOINT msg to worker") (client->identity());

  // Now send DMT_DO_CHECKPOINT message so that this process can also
  // participate in the current checkpoint
  DmtcpMessage suspendMsg(DMT_DO_CHECKPOINT);
  suspendMsg.compGroup = compId;
  client->sock() << suspendMsg;
}

bool
DmtcpCoordinator::validateNewWorkerProcess(
  DmtcpMessage &hello_remote,
  jalib::JSocket &remote,
  CoordClient *client,
  const struct sockaddr_storage *remoteAddr,
  socklen_t remoteLen)
{
  const struct sockaddr_in *sin = (const struct sockaddr_in *)remoteAddr;
  string remoteIP = inet_ntoa(sin->sin_addr);
  DmtcpMessage hello_local(DMT_ACCEPT);

  hello_local.virtualPid = client->virtualPid();
  ComputationStatus s = getStatus();

  JASSERT(hello_remote.state == WorkerState::RUNNING ||
          hello_remote.state == WorkerState::UNKNOWN) (hello_remote.state);

  if (workersRunningAndSuspendMsgSent == true) {
    /* Worker trying to connect after SUSPEND message has been sent.
     * This happens if the worker process is executing a fork() system call
     * when the DMT_DO_SUSPEND is broadcast. We need to make sure that the
     * child process is allowed to participate in the current checkpoint.
     */
    JASSERT(s.numPeers > 0) (s.numPeers);
    JASSERT(s.minimumState != WorkerState::SUSPENDED) (s.minimumState);

    // Handshake
    hello_local.compGroup = compId;
    remote << hello_local;

    // Now send DMT_DO_CHECKPOINT message so that this process can also
    // participate in the current checkpoint
    // TODO(Kapil): Make sure to walk the new worker through all the
    // already-processed pre-suspend barriers.
    DmtcpMessage suspendMsg(DMT_DO_CHECKPOINT);
    suspendMsg.compGroup = compId;
    remote << suspendMsg;
  } else if (s.numPeers > 0 && s.minimumState != WorkerState::RUNNING &&
             s.minimumState != WorkerState::UNKNOWN) {
    // If some of the processes are not in RUNNING state
    JNOTE("Current computation not in RUNNING state."
          "  Refusing to accept new connections.")
      (compId) (hello_remote.from)
      (s.numPeers) (s.minimumState);
    hello_local.type = DMT_REJECT_NOT_RUNNING;
    remote << hello_local;
    remote.close();
    return false;
  } else if (hello_remote.compGroup != UniquePid()) {
    // New Process trying to connect to Coordinator but already has compGroup
    JNOTE("New process not part of currently running computation group"
          "on this coordinator.  Rejecting.")
      (hello_remote.compGroup);

    hello_local.type = DMT_REJECT_WRONG_COMP;
    remote << hello_local;
    remote.close();
    return false;
  } else {
    // If first process, create the new computation group
    if (compId == UniquePid(0, 0, 0)) {
      // Connection of new computation.
      compId = UniquePid(hello_remote.from.hostid(), client->virtualPid(),
                         hello_remote.from.time(),
                         hello_remote.from.computationGeneration());

      // Get the resolution down to 100 mili seconds.
      curTimeStamp = getCurrTimestamp();
      numPeers = -1;
      JTRACE("First process connected.  Creating new computation group.")
        (compId);
      recordEvent("Initializing-Computation");
    } else {
      JTRACE("New process connected")
        (hello_remote.from) (client->virtualPid());
    }
    hello_local.compGroup = compId;
    hello_local.coordTimeStamp = curTimeStamp;
    if (Util::strStartsWith(remoteIP.c_str(), "127.")) {
      memcpy(&hello_local.ipAddr, &localhostIPAddr, sizeof localhostIPAddr);
    } else {
      memcpy(&hello_local.ipAddr, &sin->sin_addr, sizeof localhostIPAddr);
    }
    remote << hello_local;
  }
  return true;
}

bool
DmtcpCoordinator::startCheckpoint()
{
  ComputationStatus s = getStatus();
  if (s.minimumState == WorkerState::RUNNING && s.minimumStateUnanimous
      && !workersRunningAndSuspendMsgSent) {
    uniqueCkptFilenames = false;
    time(&ckptTimeStamp);
    JTIMER_START(checkpoint);
    recordEvent("Ckpt-Start");
    _numRestartFilenames = 0;
    _restartFilenames.clear();
    _rshCmdFileNames.clear();
    _sshCmdFileNames.clear();
    compId.incrementGeneration();
    JTRACE("starting checkpoint; incrementing generation; suspending all nodes")
      (s.numPeers) (compId.computationGeneration());

    // Pass number of connected peers to all clients
    broadcastMessage(DMT_DO_CHECKPOINT);
    // On worker side, after receiving DMT_DO_CHECKPOINT, the plugin manager
    // sends out DMTCP_EVENT_PRESUSPEND followed by DMTCP_EVENT_CHECKPOINT
    // to each plugin.  The callbacks for those events may call
    // dmtcp_global_barrier(), which sends back a DMT_BARRIER msg before
    // the workers do the actual checkpoint.

    // Suspend Message has been sent but the workers are still in running
    // state.  If the coordinator receives another checkpoint request from user
    // at this point, it should fail.
    workersRunningAndSuspendMsgSent = true;
    return true;
  } else {
    if (s.numPeers > 0) {
      JTRACE("delaying checkpoint, workers not ready") (s.minimumState)
        (s.numPeers);
    }
    return false;
  }
}

void
DmtcpCoordinator::broadcastMessage(DmtcpMessageType type,
                                   size_t extraBytes,
                                   const void *extraData)
{
  DmtcpMessage msg;

  msg.type = type;
  msg.compGroup = compId;
  msg.numPeers = clients.size();
  // From DMTCP coord viewpoint, we are killing peers after ckpt.
  // From DMTCP peer viewpoint, we will exit after ckpt.
  msg.exitAfterCkpt = exitAfterCkpt || exitAfterCkptOnce;
  msg.extraBytes = extraBytes;

  if (msg.type == DMT_KILL_PEER && clients.size() > 0) {
    killInProgress = true;
  }

  JTRACE("sending message")(type);
  for (size_t i = 0; i < clients.size(); i++) {
    clients[i]->sock() << msg;
    if (extraBytes > 0) {
      clients[i]->sock().writeAll((const char *)extraData, extraBytes);
    }
  }
  workersAtCurrentBarrier = 0;
}

DmtcpCoordinator::ComputationStatus
DmtcpCoordinator::getStatus() const
{
  ComputationStatus status;
  const static WorkerState::eWorkerState INITIAL_MIN = WorkerState::_MAX;
  const static WorkerState::eWorkerState INITIAL_MAX = WorkerState::UNKNOWN;
  int min = INITIAL_MIN;
  int max = INITIAL_MAX;
  int count = 0;
  bool unanimous = true;

  for (size_t i = 0; i < clients.size(); i++) {
    WorkerState::eWorkerState cliState = clients[i]->state();
    count++;
    unanimous = unanimous && (min == cliState || min == INITIAL_MIN);
    if (cliState < min) {
      min = cliState;
    }
    if (cliState > max) {
      max = cliState;
    }
  }

  status.minimumStateUnanimous = unanimous;
  status.minimumState = (min == INITIAL_MIN ? WorkerState::UNKNOWN
                         : (WorkerState::eWorkerState)min);
  if (status.minimumState == WorkerState::RESTARTING && count < numPeers) {
    JTRACE("minimal state counted as RESTARTING but not all processes"
           " are connected yet.  So we wait.") (numPeers) (count);
    status.minimumState = WorkerState::RESTARTING;
    status.minimumStateUnanimous = false;
  }

  status.maximumState = (max == INITIAL_MAX ? WorkerState::UNKNOWN
                         : (WorkerState::eWorkerState)max);
  status.numPeers = count;
  return status;
}

static void
signalHandler(int signum)
{
  if (signum == SIGINT) {
    prog.handleUserCommand('q');
  } else if (signum == SIGALRM) {
    timerExpired = true;
  } else {
    JASSERT(false).Text("Not reached");
  }
}

static void
setupSignalHandlers()
{
  struct sigaction action;

  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  action.sa_handler = signalHandler;

  sigaction(SIGINT, &action, NULL);
  sigaction(SIGALRM, &action, NULL);
}

// This code is also copied to ssh.cpp:updateCoordHost()
static void
calcLocalAddr()
{
  char hostname[HOST_NAME_MAX];

  JASSERT(gethostname(hostname, sizeof hostname) == 0) (JASSERT_ERRNO);

  struct addrinfo *result = NULL;
  struct addrinfo *res;
  int error;
  struct addrinfo hints;

  memset(&localhostIPAddr, 0, sizeof localhostIPAddr);
  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC; // accept AF_INET and AF_INET6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  hints.ai_protocol = 0;
  hints.ai_canonname = NULL;
  hints.ai_addr = NULL;
  hints.ai_next = NULL;

  // FROM: Wikipedia:CNAME_record:
  //  When a DNS resolver encounters a CNAME record while looking for a regular
  //  resource record, it will restart the query using the canonical name
  //  instead of the original name. (If the resolver is specifically told to
  //  look for CNAME records, the canonical name (right-hand side) is returned,
  //  rather than restarting the query.)
  hints.ai_flags |= AI_CANONNAME;
  error = getaddrinfo(hostname, NULL, &hints, &result);
  hints.ai_flags ^= AI_CANONNAME;
  if (error == 0 && result) {
    // if hostname was not fully qualified with domainname, replace it with
    // canonname.  Otherwise, keep current alias returned from gethostname().
    if ( Util::strStartsWith(result->ai_canonname, hostname) &&
         result->ai_canonname[strlen(hostname)] == '.' &&
         strlen(result->ai_canonname) < sizeof(hostname) ) {
      strncpy(hostname, result->ai_canonname, sizeof hostname);
    }
    freeaddrinfo(result);
  }
  // OPTIONAL:  If we still don't have a domainname, we could resolve with DNS
  //   (similar to 'man 1 host'), but we ont't know if Internet is present.

  /* resolve the hostname into a list of addresses */
  error = getaddrinfo(hostname, NULL, &hints, &result);
  if (error == 0) {
    /* loop over all returned results and do inverse lookup */
    bool success = false;
    bool at_least_one_match = false;
    char name[NI_MAXHOST] = "";
    for (res = result; res != NULL; res = res->ai_next) {
      struct sockaddr_in *s = (struct sockaddr_in *)res->ai_addr;

      error = getnameinfo(res->ai_addr,
                          res->ai_addrlen,
                          name,
                          NI_MAXHOST,
                          NULL,
                          0,
                          0);
      if (error != 0) {
        JTRACE("getnameinfo() failed.") (gai_strerror(error));
        continue;
      } else {
        JASSERT(sizeof localhostIPAddr == sizeof s->sin_addr);
        if ( strncmp( name, hostname, sizeof hostname ) == 0 ) {
          success = true;
          memcpy(&localhostIPAddr, &s->sin_addr, sizeof s->sin_addr);
          break; // Stop here.  We found a matching hostname.
        }
        if (!at_least_one_match) { // Prefer the first match over later ones.
          at_least_one_match = true;
          memcpy(&localhostIPAddr, &s->sin_addr, sizeof s->sin_addr);
        }
      }
    }
    if (result) {
      freeaddrinfo(result);
    }
    if (at_least_one_match) {
      success = true;  // Call it a success even if hostname != name
      if ( strncmp( name, hostname, sizeof hostname ) != 0 ) {
        JTRACE("Canonical hostname different from original hostname")
              (name)(hostname);
      }
    }

    JWARNING(success) (hostname)
      .Text("Failed to find coordinator IP address.  DMTCP may fail.");
  } else {
    if (error == EAI_SYSTEM) {
      perror("getaddrinfo");
    } else {
      JTRACE("Error in getaddrinfo") (gai_strerror(error));
    }
    inet_aton("127.0.0.1", &localhostIPAddr);
  }

  coordHostname = hostname;
}

static void
resetCkptTimer()
{
  alarm(theCheckpointInterval);
}

void
DmtcpCoordinator::updateCheckpointInterval(uint32_t interval)
{
  static bool firstClient = true;

  if ((interval != DMTCPMESSAGE_SAME_CKPT_INTERVAL &&
       interval != theCheckpointInterval) ||
      firstClient) {
    int oldInterval = theCheckpointInterval;
    if (interval != DMTCPMESSAGE_SAME_CKPT_INTERVAL) {
      theCheckpointInterval = interval;
    }
    JTRACE("CheckpointInterval updated (for this computation only)")
      (oldInterval) (theCheckpointInterval);
    firstClient = false;
    resetCkptTimer();
  }
}

void
DmtcpCoordinator::eventLoop(bool daemon)
{
  struct epoll_event ev;

  epollFd = epoll_create(MAX_EVENTS);
  JASSERT(epollFd != -1) (JASSERT_ERRNO);

  ev.events = EPOLLIN;
  ev.data.ptr = listenSock;
  JASSERT(epoll_ctl(epollFd, EPOLL_CTL_ADD, listenSock->sockfd(), &ev) != -1)
    (JASSERT_ERRNO);

  if (!daemon &&

      // epoll_ctl below fails if STDIN is pointing to /dev/null.
      // Not sure why.
      jalib::Filesystem::GetDeviceName(0) != "/dev/null" &&
      jalib::Filesystem::GetDeviceName(0) != "/dev/zero" &&
      jalib::Filesystem::GetDeviceName(0) != "/dev/random") {
    ev.events = EPOLLIN;
#ifdef EPOLLRDHUP
    ev.events |= EPOLLRDHUP;
#endif // ifdef EPOLLRDHUP
    ev.data.ptr = (void *)STDIN_FILENO;
    JASSERT(epoll_ctl(epollFd, EPOLL_CTL_ADD, STDIN_FILENO, &ev) != -1)
      (JASSERT_ERRNO);
  }

  while (true) {
    // Wait until either there is some activity on client sockets, or the timer
    // has expired.
    int nfds;
    do {
      nfds = epoll_wait(epollFd, events, MAX_EVENTS, -1);
    } while (nfds < 0 && errno == EINTR && !timerExpired);


    // The ckpt timer has expired; it's time to checkpoint.
    //   NOTE:  We need minimumStateUnanimous and RUNNING, in case
    //   worker had reached 'main()' of application and paused (e.g.,
    //   under GDB), while the ckpt interval timer went off.  We want
    //   startCheckpoint() to be deferred until the worker is RUNNING.
    ComputationStatus s = getStatus();
    if (timerExpired &&
        s.minimumStateUnanimous && s.minimumState == WorkerState::RUNNING) {
      timerExpired = false;
#ifdef TIMING
      printf("Disable checkpointing by alarm!\n");
#else
	  if (checkpoint_count == 0) {
		  startCheckpoint();
	  }
	  // startCheckpoint();
	  checkpoint_count++;
#endif
      continue;
    }

    // alarm() is not always the only source of interrupts.
    // For example, any signal, including signal 0 or SIGWINCH can cause this.
    JASSERT(nfds != -1 || errno == EINTR) (JASSERT_ERRNO);

    for (int n = 0; n < nfds; ++n) {
      void *ptr = events[n].data.ptr;
      if ((events[n].events & EPOLLHUP) ||
#ifdef EPOLLRDHUP
          (events[n].events & EPOLLRDHUP) ||
#endif // ifdef EPOLLRDHUP
          (events[n].events & EPOLLERR)) {
        JASSERT(ptr != listenSock);
        if (ptr == (void *)STDIN_FILENO) {
          JASSERT(epoll_ctl(epollFd, EPOLL_CTL_DEL, STDIN_FILENO, &ev) != -1)
            (JASSERT_ERRNO);
          close(STDIN_FD);
        } else {
          onDisconnect((CoordClient *)ptr);
        }
      } else if (events[n].events & EPOLLIN) {
        if (ptr == (void *)listenSock) {
          onConnect();
        } else if (ptr == (void *)STDIN_FILENO) {
          char buf[1];
          int ret = Util::readAll(STDIN_FD, buf, sizeof(buf));
          JASSERT(ret != -1) (JASSERT_ERRNO);
          if (ret > 0) {
            handleUserCommand(buf[0]);
          } else {
            JNOTE("closing stdin");
            JASSERT(epoll_ctl(epollFd, EPOLL_CTL_DEL, STDIN_FILENO, &ev) != -1)
              (JASSERT_ERRNO);
            close(STDIN_FD);
          }
        } else {
          onData((CoordClient *)ptr);
        }
      }
    }
  }
}

void
DmtcpCoordinator::addDataSocket(CoordClient *client)
{
  struct epoll_event ev;

#ifdef EPOLLRDHUP
  ev.events = EPOLLIN | EPOLLRDHUP;
#else // ifdef EPOLLRDHUP
  ev.events = EPOLLIN;
#endif // ifdef EPOLLRDHUP
  ev.data.ptr = client;
  JASSERT(epoll_ctl(epollFd, EPOLL_CTL_ADD, client->sock().sockfd(), &ev) != -1)
    (JASSERT_ERRNO);
}

#define shift argc--; argv++

int
main(int argc, char **argv)
{
  initializeJalib();

  // parse port
  thePort = DEFAULT_PORT;
  const char *portStr = getenv(ENV_VAR_NAME_PORT);
  if (portStr == NULL) {
    portStr = getenv("DMTCP_PORT");                      // deprecated
  }
  if (portStr != NULL) {
    thePort = jalib::StringToInt(portStr);
  }

  bool daemon = false;
  bool useLogFile = false;
  string logFilename = "";
  bool quiet = false;

  char *tmpdir_arg = NULL;

  /* NOTE: The convention is that user-specified explicit runtime arguments
   *       get a higher priority than env. vars. The logFilename variable will
   *       be over-written if the coordinator was invoked with
   *       `--logfile <filename>.
   */
  if (getenv(ENV_VAR_COORD_LOGFILE)) {
    useLogFile = true;
    logFilename = getenv(ENV_VAR_COORD_LOGFILE);
  }

  shift;
  while (argc > 0) {
    string s = argv[0];
    if (s == "-h" || s == "--help") {
      printf("%s", theUsage);
      return 1;
    } else if ((s == "--version") && argc == 1) {
      printf("%s", DMTCP_VERSION_AND_COPYRIGHT_INFO);
      return 1;
    } else if (s == "-q" || s == "--quiet") {
      quiet = true;
      jassert_quiet++;
      shift;
    } else if (s == "--exit-on-last") {
      exitOnLast = true;
      shift;
    } else if (s == "--exit-after-ckpt") {
      exitAfterCkpt = true;
      shift;
    } else if (s == "--daemon") {
      daemon = true;
      shift;
    } else if (s == "--mpi") {
      mpiMode = true;
      shift;
    } else if (s == "--coord-logfile") {
      useLogFile = true;
      logFilename = argv[1];
      shift; shift;
    } else if (s == "-i" || s == "--interval") {
      setenv(ENV_VAR_CKPT_INTR, argv[1], 1);
      shift; shift;
    } else if (argv[0][0] == '-' && argv[0][1] == 'i' &&
               isdigit(argv[0][2])) { // else if -i5, for example
      setenv(ENV_VAR_CKPT_INTR, argv[0] + 2, 1);
      shift;
    } else if (argc > 1 &&
               (s == "-p" || s == "--port" || s == "--coord-port")) {
      thePort = jalib::StringToInt(argv[1]);
      shift; shift;
    } else if (argv[0][0] == '-' && argv[0][1] == 'p' &&
               isdigit(argv[0][2])) { // else if -p0, for example
      thePort = jalib::StringToInt(argv[0] + 2);
      shift;
    } else if (argc > 1 && s == "--port-file") {
      thePortFile = argv[1];
      shift; shift;
    } else if (argc > 1 && (s == "-c" || s == "--ckptdir")) {
      setenv(ENV_VAR_CHECKPOINT_DIR, argv[1], 1);
      shift; shift;
    } else if (argc > 1 && (s == "-t" || s == "--tmpdir")) {
      tmpdir_arg = argv[1];
      shift; shift;
    } else if (s == "--write-kv-data") {
      writeKvData = true;
      shift;
    } else if (argc == 1) { // last arg can be port
      char *endptr;
      long x = strtol(argv[0], &endptr, 10);
      if ((ssize_t)strlen(argv[0]) != endptr - argv[0]) {
        fprintf(stderr, theUsage, DEFAULT_PORT);
        return 1;
      } else {
        thePort = jalib::StringToInt(argv[0]);
        shift;
      }
      x++, x--; // to suppress unused variable warning
    } else {
      fprintf(stderr, theUsage, DEFAULT_PORT);
      return 1;
    }
  }

  tmpDir = Util::calcTmpDir(tmpdir_arg);
  Util::initializeLogFile(tmpDir, NULL, NULL);

  JTRACE("New DMTCP coordinator starting.")
    (UniquePid::ThisProcess());

  if (thePort < 0) {
    fprintf(stderr, theUsage, DEFAULT_PORT);
    return 1;
  }

  calcLocalAddr();

  if (getenv(ENV_VAR_CHECKPOINT_DIR) != NULL) {
    ckptDir = getenv(ENV_VAR_CHECKPOINT_DIR);
  } else {
    ckptDir = get_current_dir_name();
  }

  // Check and enable dumping KVDB.
  {
    const char *kvdbEnv = getenv(ENV_VAR_COORD_WRITE_KVDB);
    if (kvdbEnv != NULL && kvdbEnv[0] == '1') {
      writeKvData = true;
    }
  }

  /*Test if the listener socket is already open*/
  if (fcntl(PROTECTED_COORD_FD, F_GETFD) != -1) {
    listenSock = new jalib::JServerSocket(PROTECTED_COORD_FD);
    JASSERT(listenSock->port() != -1).Text("Invalid listener socket");
    JTRACE("Using already created listener socket") (listenSock->port());
  } else {
    errno = 0;
    listenSock = new jalib::JServerSocket(jalib::JSockAddr::ANY, thePort, 128);
    JASSERT(listenSock->isValid()) (thePort) (JASSERT_ERRNO)
    .Text("Failed to create listen socket."
          "\nIf msg is \"Address already in use\", "
          "this may be an old coordinator."
          "\nKill default coordinator and try again:  dmtcp_command -q"
          "\nIf that fails, \"pkill -9 dmtcp_coord\","
          " and try again in a minute or so.");
  }

  thePort = listenSock->port();
  if (!thePortFile.empty()) {
    Util::writeCoordPortToFile(thePort, thePortFile.c_str());
  }
  JTRACE("Listening on port")(thePort);

  // parse checkpoint interval
  const char *interval = getenv(ENV_VAR_CKPT_INTR);
  if (interval != NULL) {
    theDefaultCheckpointInterval = jalib::StringToInt(interval);
    theCheckpointInterval = theDefaultCheckpointInterval;
  }

#if 0
  if (!quiet) {
    JASSERT_STDERR <<
      "dmtcp_coordinator starting..." <<
      "\n    Port: " << thePort <<
      "\n    Checkpoint Interval: ";
    if (theCheckpointInterval == 0) {
      JASSERT_STDERR << "disabled (checkpoint manually instead)";
    } else {
      JASSERT_STDERR << theCheckpointInterval;
    }
    JASSERT_STDERR <<
      "\n    Exit on last client: " << exitOnLast << "\n";
  }
#else // if 0
  if (!quiet) {
    fprintf(stderr, "dmtcp_coordinator starting..."
                    "\n    Host: %s (%s)"
                    "\n    Port: %d"
                    "\n    Checkpoint Interval: ",
            coordHostname.c_str(), inet_ntoa(localhostIPAddr), thePort);
    if (theCheckpointInterval == 0) {
      fprintf(stderr, "disabled (checkpoint manually instead)");
    } else {
      fprintf(stderr, "%d", theCheckpointInterval);
    }
    fprintf(stderr, "\n    Exit on last client: %d\n", exitOnLast);
  }
#endif // if 0

  if (daemon) {
    if (!quiet) {
      JASSERT_STDERR << "Backgrounding...\n";
    }
    int fd = -1;
    if (!useLogFile) {
      fd = open("/dev/null", O_RDWR);
      JASSERT(dup2(fd, STDIN_FILENO) == STDIN_FILENO);
    } else {
      fd = open(logFilename.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0666);
      JASSERT_SET_LOG(logFilename, "", "");
      int nullFd = open("/dev/null", O_RDWR);
      JASSERT(dup2(nullFd, STDIN_FILENO) == STDIN_FILENO);
      close(nullFd);
    }
    JASSERT(dup2(fd, STDOUT_FILENO) == STDOUT_FILENO);
    JASSERT(dup2(fd, STDERR_FILENO) == STDERR_FILENO);
    JASSERT_CLOSE_STDERR();
    if (fd > STDERR_FILENO) {
      close(fd);
    }

    if (fork() > 0) {
      JTRACE("Parent Exiting after fork()");
      exit(0);
    }

    // pid_t sid = setsid();
  } else {
    if (!quiet) {
      JASSERT_STDERR <<
        "Type '?' for help." <<
        "\n\n";
    }
  }

  /* We set up the signal handler for SIGINT and SIGALRM.
   * SIGINT is used to send DMT_KILL_PEER message to all the connected peers
   * before exiting.
   * SIGALRM is used for interval checkpointing.
   */
  setupSignalHandlers();

  /* If the coordinator was started transparently by dmtcp_launch, then we
   * want to block signals, such as SIGINT.  To see why this is important:
   * % gdb dmtcp_launch a.out
   * (gdb) run
   * ^C   # Stop gdb to get its attention, and continue debugging.
   * # The above scenario causes the SIGINT to go to a.out and its child,
   * # the dmtcp_coordinator.  The coord then triggers the SIGINT handler,
   * # which sends DMT_KILL_PEER to kill a.out.
   */
  if (exitOnLast && daemon) {
    sigset_t set;
    sigfillset(&set);

    // unblock SIGALRM because we are using alarm() for interval checkpointing
    sigdelset(&set, SIGALRM);

    // sigprocmask is only per-thread; but the coordinator is single-threaded.
    sigprocmask(SIG_BLOCK, &set, NULL);
  }

  if (mpiMode) {
    prog.writeSubmissionHostInfo();
  }
  prog.eventLoop(daemon);
  return 0;
}
